#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ FULL DROP-IN: Manhattan routes + robust continuum heatmap ============
# What this does:
# 1) reads + aligns data, clips roads to Manhattan (EPSG:2263)
# 2) builds graph with 25 mph edge weights (min) and writes nodes/edges
# 3) solves node potentials using distance-to-destination (source only, no sink)
# 4) computes Greedy + Dijkstra routes; writes GeoJSON and PNG
# 5) enriches Voronoi with Kxx/Kxy/Kyy and writes GeoJSON + centroids CSV
# 6) solves continuum PDE with anisotropic K and plots a heatmap (no NaN rims)
# ================================================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from collections import defaultdict
from shapely.validation import make_valid
from shapely.geometry import LineString, Point
from shapely import union_all
from shapely.ops import substring

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csgraph
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import Delaunay

# ===========================================================================
# CONFIGURATION
# ===========================================================================

TARGET_CRS = "EPSG:2263"  # NYC State Plane (feet)

# ----------------------- REQUIRED INPUT FILES (in data/) -----------------------
ROADS_IN = "data/filtered_roads.geojson"         # Full NYC road network
MAN_IN   = "data/manhattan_boundary.geojson"     # Manhattan boundary polygon
VORO_IN  = "data/singum_voronoi.geojson"         # Voronoi cells (from singum.py)

# ----------------------- GENERATED OUTPUT FILES (in output/) -----------------------
# Intermediate files
ROADS_2263 = "output/filtered_roads_2263.geojson"           # Roads reprojected to EPSG:2263
MAN_2263   = "output/manhattan_boundary_2263.geojson"       # Boundary reprojected to EPSG:2263
ROADS_CLIP = "output/filtered_roads_manhattan_2263.geojson" # Roads clipped to Manhattan
GRAPH_TXT  = "output/graph_edges.txt"                       # Graph adjacency list
NODES_GDF  = "output/nodes_points_2263.geojson"             # Node coordinates
NODES_PHI  = "output/nodes_with_phi_2263.geojson"           # Nodes with potential values

# Route outputs
ROUTES_GEOJSON    = "output/manhattan_routes.geojson"       # Route geometries
ROUTES_PNG        = "output/manhattan_routes_viz.png"       # Greedy vs Dijkstra comparison
ROUTES_SINGUM_PNG = "output/manhattan_routes_singum_only.png"

# Voronoi/K tensor outputs
VORO_OUT       = "output/singum_voronoi_with_K.geojson"     # Voronoi with K tensors
VORO_PNG       = "output/singum_voronoi.png"                # Voronoi visualization
K_CENTROID_CSV = "output/K_centroids_2263.csv"              # K tensor centroids (optional)

# Continuum outputs
HEATMAP_PNG = "output/manhattan_continuum_V.png"            # Continuum potential heatmap

#landmark locations
COLUMBIA_LONLAT      = (-73.9626, 40.8075)
MSG_LONLAT           = (-73.9934, 40.7505)  # Madison Square Garden
GRAND_CENTRAL_LONLAT = (-73.9772, 40.7527)
NYU_LONLAT           = (-73.9965, 40.7295)
TIMES_SQUARE_LONLAT  = (-73.9855, 40.7580)
PENN_STATION_LONLAT  = (-73.9937, 40.7506)
WALL_STREET_LONLAT   = (-74.0110, 40.7074)
GW_BRIDGE_LONLAT     = (-73.9527, 40.8517)  # George Washington Bridge

DESTINATIONS = {
    "Grand Central": GRAND_CENTRAL_LONLAT,
    "Columbia": COLUMBIA_LONLAT,
    "NYU": NYU_LONLAT,
    "Times Square": TIMES_SQUARE_LONLAT,
    "Wall Street": WALL_STREET_LONLAT,
}

# SCENARIO PARAMETERS

# Fig 5c/5d scenario settings
MSG_RADIUS_FT    = 400.0   # Radius for MSG source/sink region
COLUMBIA_RADIUS_FT = 600.0
GW_COMMUTERS     = 5000    # Fig 5d: commuters from GW Bridge
NBA_ATTENDEES    = 20000   # Fig 5c: people leaving MSG

# Continuum PDE settings
TOTAL_FLUX_INTO_COLUMBIA = 5000.0
GRID_NX, GRID_NY = 700, 1050  # Continuum grid resolution
GRID_BUFFER_FT   = 250.0      # Buffer around Manhattan for grid

# These are only used by the basic main() pipeline, not Fig 5c/5d
START_NODEID = 21465  # Default source node ID
END_NODEID   = 78136  # Default destination node ID

MPH_TO_FTPS = 1.46667   # mph to feet per second
SEC_TO_MIN  = 1.0/60.0  # seconds to minutes
CONST_MPH   = 25.0      # Default speed limit (mph)

# helper functions
def nan_gaussian_blur(Z, sigma_pix=0.8):
    """NaN-aware Gaussian blur for pretty fields."""
    from scipy.signal import convolve2d
    if Z is None or sigma_pix <= 0:
        return Z
    wrad = int(np.ceil(3*sigma_pix))
    y, x = np.mgrid[-wrad:wrad+1, -wrad:wrad+1]
    G = np.exp(-(x**2+y**2)/(2*sigma_pix**2)); G /= G.sum()
    M = np.isfinite(Z).astype(float)
    Z0 = np.where(np.isfinite(Z), Z, 0.0)
    num = convolve2d(Z0, G, mode='same', boundary='symm')
    den = convolve2d(M,  G, mode='same', boundary='symm')
    out = num / np.maximum(den, np.finfo(float).eps)
    out[den==0] = np.nan
    return out

def apply_dirichlet_row_replace(M, nodes, value):
    ML = M.tolil(copy=True); b = np.zeros(M.shape[0], float)
    for n in np.atleast_1d(nodes).astype(int):
        ML.rows[n] = [n]
        ML.data[n] = [1.0]
        b[n] = value
    return ML.tocsr(), b

def solve_with_component_pinning(M, b, pinned_nodes):
    A = M.copy().tocsr()
    A.setdiag(0); A.eliminate_zeros()
    A = (A != 0).astype(int)
    n_comp, labels = csgraph.connected_components(A, directed=False)

    pinned_labels = set(labels[np.atleast_1d(pinned_nodes)])
    ML = M.tolil(copy=True)
    for comp in range(n_comp):
        if comp in pinned_labels:
            continue
        n = int(np.where(labels == comp)[0][0])
        ML.rows[n] = [n]; ML.data[n] = [1.0]; b[n] = 0.0
    return spsolve(ML.tocsr(), b)

# ============================================================================
# FEM SOLVER FUNCTIONS (P1 triangular elements with anisotropic diffusion)
# ============================================================================

def _create_fem_mesh(X, Y, inside):
    """
    Create triangular mesh from grid points inside the domain.

    Returns:
        nodes: (N, 2) array of node coordinates
        elements: (M, 3) array of triangle vertex indices
        grid_to_fem: mapping from grid index (j*nx + i) to FEM node index (-1 if outside)
        fem_to_grid: mapping from FEM node index to grid index (j, i)
    """
    nxg, nyg = X.shape[1], X.shape[0]

    # Extract interior points
    interior_indices = []
    grid_to_fem = -np.ones(nxg * nyg, dtype=int)
    fem_to_grid = []

    node_idx = 0
    for j in range(nyg):
        for i in range(nxg):
            if inside[j, i]:
                grid_idx = j * nxg + i
                grid_to_fem[grid_idx] = node_idx
                fem_to_grid.append((j, i))
                interior_indices.append(grid_idx)
                node_idx += 1

    # Get coordinates of interior nodes
    nodes = np.column_stack([X.flatten()[interior_indices],
                              Y.flatten()[interior_indices]])

    # Create Delaunay triangulation
    tri = Delaunay(nodes)
    elements = tri.simplices

    return nodes, elements, grid_to_fem, fem_to_grid


def _element_stiffness_anisotropic(coords, K):
    """
    Compute element stiffness matrix for a P1 triangle with anisotropic diffusion.

    Weak form: ∫ (K ∇φ) · ∇v dΩ

    Args:
        coords: (3, 2) array of triangle vertex coordinates
        K: (2, 2) conductivity tensor at element centroid

    Returns:
        Ke: (3, 3) element stiffness matrix
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Area of triangle (signed)
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    if abs(area) < 1e-12:
        return np.zeros((3, 3))

    # Gradient of basis functions (constant on P1 element)
    # N_i = (a_i + b_i*x + c_i*y) / (2*area)
    # ∇N_i = [b_i, c_i] / (2*area)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Gradients: grad_N[i] = [b[i], c[i]] / (2*area)
    inv_2area = 1.0 / (2.0 * area)

    # Element stiffness: Ke[i,j] = ∫ (K ∇N_i) · ∇N_j dΩ
    #                            = |area| * (K ∇N_i) · ∇N_j
    Ke = np.zeros((3, 3))
    for i in range(3):
        grad_i = np.array([b[i], c[i]]) * inv_2area
        for j in range(3):
            grad_j = np.array([b[j], c[j]]) * inv_2area
            # (K ∇N_j) · ∇N_i
            Ke[i, j] = abs(area) * np.dot(K @ grad_j, grad_i)

    return Ke


def _assemble_fem_system(nodes, elements, K_func):
    """
    Assemble global FEM stiffness matrix for anisotropic diffusion.

    Args:
        nodes: (N, 2) array of node coordinates
        elements: (M, 3) array of triangle vertex indices
        K_func: function(x, y) -> (2, 2) conductivity tensor

    Returns:
        A: (N, N) sparse stiffness matrix
    """
    N = len(nodes)
    A = lil_matrix((N, N))

    for elem in elements:
        # Get element coordinates
        coords = nodes[elem]

        # Get K at centroid
        centroid = coords.mean(axis=0)
        K = K_func(centroid[0], centroid[1])

        # Skip degenerate elements or invalid K
        if K is None or not np.all(np.isfinite(K)):
            continue

        # Compute element stiffness
        Ke = _element_stiffness_anisotropic(coords, K)

        # Assemble into global matrix
        for i_local in range(3):
            i_global = elem[i_local]
            for j_local in range(3):
                j_global = elem[j_local]
                A[i_global, j_global] += Ke[i_local, j_local]

    return A.tocsr()


def solve_fem_anisotropic(X, Y, inside, Kxxg, Kxyg, Kyyg,
                           dirichlet_masks, dirichlet_values,
                           neumann_mask=None, neumann_flux=0.0):
    """
    Solve anisotropic diffusion PDE using P1 FEM:
        ∇·(K∇φ) = 0

    Args:
        X, Y: meshgrid arrays
        inside: boolean mask of interior points
        Kxxg, Kxyg, Kyyg: K tensor components on grid
        dirichlet_masks: list of (mask, value) tuples for Dirichlet BCs
        dirichlet_values: corresponding values (list aligned with masks)
        neumann_mask: optional mask for Neumann BC nodes
        neumann_flux: flux value for Neumann BC

    Returns:
        V: solution array on original grid (NaN outside domain)
    """
    print("Creating FEM mesh from grid points...")
    nodes, elements, grid_to_fem, fem_to_grid = _create_fem_mesh(X, Y, inside)
    N_fem = len(nodes)
    nxg, nyg = X.shape[1], X.shape[0]

    print(f"FEM mesh: {N_fem} nodes, {len(elements)} triangular elements")

    # Create K interpolation function from grid
    def K_at_point(x, y):
        # Find nearest grid point
        i = int(np.clip(np.round((x - X[0, 0]) / (X[0, 1] - X[0, 0])), 0, nxg - 1))
        j = int(np.clip(np.round((y - Y[0, 0]) / (Y[1, 0] - Y[0, 0])), 0, nyg - 1))

        kxx = Kxxg[j, i]
        kxy = Kxyg[j, i]
        kyy = Kyyg[j, i]

        if not (np.isfinite(kxx) and np.isfinite(kxy) and np.isfinite(kyy)):
            # Default to isotropic if invalid
            return np.eye(2) * 1.0

        return np.array([[kxx, kxy], [kxy, kyy]])

    print("Assembling FEM stiffness matrix...")
    A = _assemble_fem_system(nodes, elements, K_at_point)
    b = np.zeros(N_fem)

    # Apply Dirichlet boundary conditions
    dirichlet_nodes = set()
    A_lil = A.tolil()

    for mask, value in zip(dirichlet_masks, dirichlet_values):
        for j in range(nyg):
            for i in range(nxg):
                if mask[j, i]:
                    grid_idx = j * nxg + i
                    fem_idx = grid_to_fem[grid_idx]
                    if fem_idx >= 0:
                        dirichlet_nodes.add(fem_idx)
                        # Row replacement for Dirichlet BC
                        A_lil[fem_idx, :] = 0
                        A_lil[fem_idx, fem_idx] = 1.0
                        b[fem_idx] = value

    # Small diagonal regularization for stability
    A = A_lil.tocsr()
    eps = 1e-10
    diag_reg = csr_matrix((np.full(N_fem, eps), (np.arange(N_fem), np.arange(N_fem))), shape=(N_fem, N_fem))
    A = A + diag_reg

    # Sanitize
    if not np.isfinite(A.data).all():
        A.data[~np.isfinite(A.data)] = 0.0
    if not np.isfinite(b).all():
        b[~np.isfinite(b)] = 0.0

    print("Solving FEM system...")
    phi_fem = spsolve(A, b)
    print("FEM solve complete.")

    # Map solution back to grid
    V = np.full(X.shape, np.nan)
    for fem_idx, (j, i) in enumerate(fem_to_grid):
        V[j, i] = phi_fem[fem_idx]

    return V


def greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=None, max_steps=500000, sink_nodes=None):
    """Forward greedy gradient descent from src to dst using directed edges.

    If sink_nodes is provided, stops when reaching any sink node (not just dst).
    """
    from collections import defaultdict
    # Build directed adjacency list (respect one-way streets)
    nbrs = defaultdict(list)
    for u, v in zip(efrom, eto):
        nbrs[u].append(v)  # From u we can go to v

    goal = int(dst)
    tol = 1e-12
    cur = int(src)
    path = [cur]
    visited = {cur}

    # If sink_nodes provided, use it; otherwise just check for dst
    if sink_nodes is None:
        sink_nodes = {goal}
    else:
        sink_nodes = set(sink_nodes) | {goal}

    def dist_goal(u):
        return 0.0 if coords is None else np.linalg.norm(coords[u] - coords[goal])

    steps = 0
    while cur not in sink_nodes and steps < max_steps:
        steps += 1
        nlist = nbrs[cur]

        # Filter to unvisited neighbors only (strict no-revisit)
        unvisited = [v for v in nlist if v not in visited]

        # Among unvisited, prefer lower phi
        lower = [v for v in unvisited if phi[v] < phi[cur] - tol]
        if lower:
            best_phi = min(phi[v] for v in lower)
            candidates = [v for v in lower if abs(phi[v] - best_phi) <= tol]
            if len(candidates) > 1:
                nxt = min(candidates, key=dist_goal)
            else:
                nxt = candidates[0]
        else:
            # No lower unvisited, try equal phi unvisited
            equal = [v for v in unvisited if abs(phi[v] - phi[cur]) <= tol]
            if equal:
                nxt = min(equal, key=dist_goal)
            else:
                # No good unvisited options, try any unvisited
                if unvisited:
                    nxt = min(unvisited, key=lambda v: (phi[v], dist_goal(v)))
                else:
                    # All neighbors visited - stuck
                    break

        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return path


def reverse_greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=None, max_steps=500000,
                                  goal_weight=0.3):
    from collections import defaultdict

    # Build directed adjacency list (respect one-way streets)
    # Since we trace backwards (dst->src), we need reverse edges: who can reach node v?
    # An edge u->v means v can be reached from u, so in reverse: from v we can go to u
    reverse_nbrs = defaultdict(list)
    for u, v in zip(efrom, eto):
        reverse_nbrs[v].append(u)  # v can go backwards to u

    # Start at dst, goal is src (reverse direction)
    goal = int(src)
    cur = int(dst)
    path = [cur]
    visited = {cur}

    tol = 1e-12

    def dist_goal(u):
        return 0.0 if coords is None else np.linalg.norm(coords[u] - coords[goal])

    initial_dist = dist_goal(cur)

    steps = 0
    while cur != goal and steps < max_steps:
        steps += 1
        nlist = reverse_nbrs[cur]

        if not nlist:
            break

        phi_cur = phi[cur]

        # Score candidates: prefer higher phi (uphill) with goal heuristic
        scored_candidates = []
        for v in nlist:
            if v in visited and v != goal:
                continue

            delta_phi = phi[v] - phi_cur  # positive = uphill (good)

            # Normalize delta_phi to [0, 1] range approximately
            # Use phi range as reference
            phi_range = max(phi.max() - phi.min(), 1e-12)
            norm_delta = delta_phi / phi_range

            # Goal direction score (higher = closer to goal)
            if coords is not None and initial_dist > tol:
                d_cur = dist_goal(cur)
                d_next = dist_goal(v)
                # Progress toward goal (positive = getting closer)
                progress = (d_cur - d_next) / initial_dist
            else:
                progress = 0.0

            # Combined score: blend gradient ascent with goal heuristic
            # Higher score = better candidate
            score = (1 - goal_weight) * norm_delta + goal_weight * progress

            scored_candidates.append((v, score, delta_phi, dist_goal(v)))

        if not scored_candidates:
            # All neighbors visited, allow backtracking to any neighbor
            for v in reverse_nbrs[cur]:
                if v == goal:
                    path.append(v)
                    return path[::-1]
                delta_phi = phi[v] - phi_cur
                scored_candidates.append((v, delta_phi, delta_phi, dist_goal(v)))

        if not scored_candidates:
            break

        # Sort by score (descending), then by distance to goal (ascending) as tiebreaker
        scored_candidates.sort(key=lambda x: (-x[1], x[3]))

        # Pick best candidate
        nxt = scored_candidates[0][0]

        # Avoid immediate backtracking if possible
        if len(path) >= 2 and nxt == path[-2] and len(scored_candidates) > 1:
            nxt = scored_candidates[1][0]

        if nxt == goal:
            path.append(nxt)
            break

        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    # Reverse the path so it goes src -> dst
    return path[::-1]


def path_time(path, time_map):
    """
    Compute total travel time for a path.

    Note: Only looks up edges in the forward direction (a→b).
    Does not fall back to reverse direction to ensure path validity.
    """
    if len(path) < 2:
        return np.nan
    t = 0.0
    for a, b in zip(path[:-1], path[1:]):
        w = time_map.get((a, b))
        if w is None:
            return np.nan
        t += w
    return t

def lonlat_to_2263(lonlat):
    """Convert (lon, lat) to EPSG:2263 (ft)."""
    man_crs = gpd.read_file(MAN_2263).crs
    return gpd.GeoSeries([Point(lonlat)], crs="EPSG:4326").to_crs(man_crs).iloc[0]

def nearest_node(coords, xy):
    """Find nearest node index to a given (x, y) coordinate."""
    return int(np.argmin(np.linalg.norm(coords - np.array(xy)[None,:], axis=1)))

def dijkstra_path(coords, efrom, eto, etime, src, dst):
    """
    Find shortest path using Dijkstra's algorithm.

    Parameters:
    -----------
    coords : array
        Node coordinates (N x 2)
    efrom, eto : arrays
        Edge endpoint indices
    etime : array
        Edge travel times
    src, dst : int
        Source and destination node indices

    Returns:
    --------
    path : list
        Node indices from src to dst
    time : float
        Total travel time
    """
    rows = np.concatenate([efrom, eto])
    cols = np.concatenate([eto, efrom])
    dat = np.concatenate([etime, etime])
    N = coords.shape[0]
    adj = csr_matrix((dat, (rows, cols)), shape=(N, N))
    dist, preds = csgraph.dijkstra(adj, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]):
        return [], np.inf
    path = []
    cur = dst
    while cur != -9999 and cur != src:
        path.append(cur)
        cur = preds[cur]
    path.append(src)
    path = path[::-1]
    return path, float(dist[dst])

def compute_shortest_path_potential(efrom, eto, etime, target, V):
    """
    Compute shortest-path distances from all nodes to a target node.

    This creates a potential field where phi[n] = distance from node n to target.
    Used for greedy routing: following decreasing phi leads to target.

    Parameters:
    -----------
    efrom, eto : arrays
        Edge endpoint indices
    etime : array
        Edge travel times
    target : int
        Target node index (destination)
    V : int
        Number of nodes

    Returns:
    --------
    phi : array
        Potential values (distances to target) for each node
    """
    rows = np.concatenate([efrom, eto])
    cols = np.concatenate([eto, efrom])
    dat = np.concatenate([etime, etime])
    adj = csr_matrix((dat, (rows, cols)), shape=(V, V))
    # Transpose to get distances TO target (not FROM target)
    adj_T = adj.T.tocsr()
    dist = csgraph.dijkstra(adj_T, directed=True, indices=target)
    # Handle unreachable nodes
    max_dist = np.nanmax(dist[np.isfinite(dist)]) if np.any(np.isfinite(dist)) else 0.0
    dist[~np.isfinite(dist)] = max_dist
    return dist

def reverse_greedy_from_sink(phi, efrom, eto, src, dst, coords=None, max_steps=500000):
    """
    Reverse greedy search: trace from destination (sink) back to source.

    Uses potential phi = distance FROM source, so:
    - phi[source] = 0 (low)
    - phi[destination] = high

    Traces backwards using REVERSE edges, following DECREASING phi toward source.
    Returns path in forward order (src -> dst).

    Parameters:
    -----------
    phi : array
        Node potentials (distance from source; source=0, destination=high)
    efrom, eto : arrays
        Edge endpoints (forward direction)
    src, dst : int
        Source and destination node indices
    coords : array, optional
        Node coordinates for tiebreaking
    max_steps : int
        Maximum steps before termination

    Returns:
    --------
    path : list
        Node indices from src to dst
    """
    from collections import defaultdict

    # Build REVERSE adjacency: who can reach each node?
    # Edge u->v means from v we can trace back to u
    reverse_nbrs = defaultdict(list)
    for u, v in zip(efrom, eto):
        reverse_nbrs[v].append(u)

    goal = int(src)
    cur = int(dst)
    path = [cur]
    visited = {cur}
    tol = 1e-12

    def dist_goal(u):
        return 0.0 if coords is None else np.linalg.norm(coords[u] - coords[goal])

    steps = 0
    while cur != goal and steps < max_steps:
        steps += 1
        nlist = reverse_nbrs[cur]

        if not nlist:
            break

        # Find neighbors with LOWER phi (heading toward source where phi=0)
        lower = [v for v in nlist if phi[v] < phi[cur] - tol and v not in visited]

        if lower:
            # Pick the one with lowest phi (closest to source)
            best_phi = min(phi[v] for v in lower)
            candidates = [v for v in lower if abs(phi[v] - best_phi) <= tol]
            if len(candidates) > 1:
                nxt = min(candidates, key=dist_goal)
            else:
                nxt = candidates[0]
        else:
            # No unvisited lower neighbors, try equal phi
            equal = [v for v in nlist if abs(phi[v] - phi[cur]) <= tol and v not in visited]
            if equal:
                nxt = min(equal, key=dist_goal)
            else:
                # Stuck - no valid moves
                break

        if nxt == goal:
            path.append(nxt)
            break

        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    # Reverse path so it goes src -> dst
    return path[::-1]

def gradient_trace(gradx_interp, grady_interp, coords, src, dst, steplength=50.0, max_steps=50000,
                   initial_direction=None, initial_kick_steps=5, goal_blend_weight=0.3,
                   use_goal_directed=True):
    """
    Follow continuous -grad(phi) to trace a path from src to dst.
    Adapted for Manhattan scale (feet).

    Uses GOAL-DIRECTED gradient following: blends the gradient direction with
    direct-to-goal direction to prevent getting stuck when gradient is weak,
    noisy, or pointing the wrong way (common near source regions).

    Parameters:
    -----------
    initial_direction : array-like or None
        If provided, use this direction for the first `initial_kick_steps` steps
        before switching to gradient following. This helps when the gradient
        at the source is ambiguous (e.g., at a source point where phi is maximum).
    initial_kick_steps : int
        Number of steps to follow the initial_direction before using gradient.
    goal_blend_weight : float
        Weight for goal direction in blending (0 = pure gradient, 1 = pure goal-seeking).
        Higher values make the path more direct but less influenced by the potential field.
    use_goal_directed : bool
        If True, uses adaptive goal-directed blending. If False, pure gradient following.
    """
    from scipy.interpolate import RegularGridInterpolator

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    pt = coords[int(src)].astype(float).copy()
    goal = coords[int(dst)].astype(float)
    path_xy = [pt.copy()]

    initial_dist = np.linalg.norm(goal - pt)

    # Normalize initial direction if provided
    init_dir = None
    if initial_direction is not None:
        init_dir = np.array(initial_direction, float)
        norm = np.linalg.norm(init_dir)
        if norm > 1e-12:
            init_dir = init_dir / norm
        else:
            init_dir = None

    stuck_counter = 0
    prev_dist_to_goal = initial_dist

    for step_idx in range(int(max_steps)):
        dist_to_goal = np.linalg.norm(goal - pt)

        # Check if we're close enough to goal
        if dist_to_goal <= 2 * steplength:
            path_xy.append(goal.copy())
            break

        # Use initial direction for first few steps (Dijkstra kick)
        if init_dir is not None and step_idx < initial_kick_steps:
            pt = pt + steplength * init_dir
        else:
            gx = gradx_interp((pt[1], pt[0]))
            gy = grady_interp((pt[1], pt[0]))

            # Direction toward goal (always valid)
            dir_goal = goal - pt
            dir_goal_norm = np.linalg.norm(dir_goal)
            if dir_goal_norm > 1e-12:
                dir_goal = dir_goal / dir_goal_norm
            else:
                path_xy.append(goal.copy())
                break

            # Handle invalid gradient
            if not np.isfinite(gx) or not np.isfinite(gy):
                # Fall back to goal direction
                pt = pt + steplength * dir_goal
            else:
                gvec = np.array([gx, gy], float)
                gnrm = np.linalg.norm(gvec)

                if gnrm < 1e-12:
                    # Near zero gradient, head directly to goal
                    pt = pt + steplength * dir_goal
                else:
                    # Negative gradient direction (should point toward lower phi)
                    neg_grad = -gvec / gnrm

                    if use_goal_directed:
                        # Check if gradient is pointing toward or away from goal
                        grad_goal_alignment = np.dot(neg_grad, dir_goal)

                        # Adaptive blending based on:
                        # 1. Gradient-goal alignment (blend more if gradient opposes goal)
                        # 2. Distance from source (blend more when near source)
                        # 3. Progress stall detection

                        # If gradient opposes goal direction, increase goal weight
                        if grad_goal_alignment < 0:
                            # Gradient pointing AWAY from goal - this is the problem case
                            # Use stronger goal direction
                            adaptive_weight = min(0.8, goal_blend_weight + 0.4 * abs(grad_goal_alignment))
                        else:
                            # Gradient roughly aligned with goal - trust it more
                            # But still blend a bit for robustness
                            adaptive_weight = goal_blend_weight * (1 - grad_goal_alignment)

                        # Near-source region: rely more on goal direction
                        # (gradient is ambiguous near Dirichlet BC at source)
                        progress_ratio = dist_to_goal / max(initial_dist, 1e-12)
                        if progress_ratio > 0.9:  # Still near source (< 10% progress)
                            adaptive_weight = max(adaptive_weight, 0.5)

                        # Stuck detection: if not making progress, increase goal weight
                        if dist_to_goal >= prev_dist_to_goal - steplength * 0.1:
                            stuck_counter += 1
                            if stuck_counter > 10:
                                adaptive_weight = min(0.9, adaptive_weight + 0.1 * (stuck_counter - 10))
                        else:
                            stuck_counter = max(0, stuck_counter - 1)

                        # Blend gradient and goal directions
                        blended_dir = (1 - adaptive_weight) * neg_grad + adaptive_weight * dir_goal
                        blended_norm = np.linalg.norm(blended_dir)
                        if blended_norm > 1e-12:
                            blended_dir = blended_dir / blended_norm
                        else:
                            blended_dir = dir_goal

                        pt = pt + steplength * blended_dir
                    else:
                        # Pure gradient following (original behavior)
                        pt = pt - steplength * (gvec / gnrm)

        prev_dist_to_goal = dist_to_goal

        pt[0] = np.clip(pt[0], x_min, x_max)
        pt[1] = np.clip(pt[1], y_min, y_max)
        path_xy.append(pt.copy())

        # Safety: if we've gone way past max distance, something is wrong
        if np.linalg.norm(pt - coords[int(src)]) > 3 * initial_dist:
            break

    return np.vstack(path_xy) if path_xy else np.zeros((0, 2))

def snap_path_to_network(path_xy, coords, efrom, eto, src, dst):
    """
    Snap a continuous gradient-traced path onto the physical network.
    """
    from collections import defaultdict, deque

    if len(path_xy) < 2:
        return [src, dst] if src != dst else [src]

    # Build adjacency list
    adj = defaultdict(list)
    for u, v in zip(efrom, eto):
        adj[u].append(v)
        adj[v].append(u)

    node_path = [src]
    current = src
    visited = {src}

    max_steps = len(coords)
    step_count = 0

    while current != dst and step_count < max_steps:
        step_count += 1
        neighbors = adj[current]
        if not neighbors:
            break

        current_pos = coords[current]
        dst_pos = coords[dst]

        # Find closest point on continuous path
        dists = np.linalg.norm(path_xy - current_pos, axis=1)
        closest_idx = np.argmin(dists)

        # Look ahead on path
        lookahead_idx = min(closest_idx + 10, len(path_xy) - 1)
        target_pos = path_xy[lookahead_idx]

        desired_dir = target_pos - current_pos
        if np.linalg.norm(desired_dir) > 1e-12:
            desired_dir = desired_dir / np.linalg.norm(desired_dir)

        best_neighbor = None
        best_score = -np.inf
        current_dist_to_dst = np.linalg.norm(current_pos - dst_pos)

        for nb in neighbors:
            if nb in visited and nb != dst:
                continue

            nb_pos = coords[nb]
            edge_dir = nb_pos - current_pos
            edge_len = np.linalg.norm(edge_dir)
            if edge_len < 1e-12:
                continue
            edge_dir = edge_dir / edge_len

            alignment = np.dot(edge_dir, desired_dir)
            nb_dist_to_dst = np.linalg.norm(nb_pos - dst_pos)
            progress = current_dist_to_dst - nb_dist_to_dst

            score = alignment + 0.3 * (progress / max(edge_len, 1e-12))
            if nb == dst:
                score += 10.0

            if score > best_score:
                best_score = score
                best_neighbor = nb

        if best_neighbor is None:
            for nb in neighbors:
                if nb not in visited:
                    best_neighbor = nb
                    break

        if best_neighbor is None:
            break

        node_path.append(best_neighbor)
        visited.add(best_neighbor)
        current = best_neighbor

    # If didn't reach dst, BFS to connect
    if current != dst:
        queue = deque([(current, [current])])
        bfs_visited = {current}
        found_path = None

        while queue and found_path is None:
            node, path = queue.popleft()
            for nb in adj[node]:
                if nb == dst:
                    found_path = path + [dst]
                    break
                if nb not in bfs_visited:
                    bfs_visited.add(nb)
                    queue.append((nb, path + [nb]))

        if found_path and len(found_path) > 1:
            node_path.extend(found_path[1:])

    return node_path

# DATA PREP
def align_and_clip():
    roads = gpd.read_file(ROADS_IN)
    man   = gpd.read_file(MAN_IN)

    # Align CRS
    if man.crs is None and roads.crs is not None:
        man = man.set_crs(roads.crs)
    if roads.crs is None:
        raise RuntimeError("Roads file has no CRS.")
    if roads.crs.to_string() != TARGET_CRS:
        roads = roads.to_crs(TARGET_CRS)
    if man.crs is None or man.crs.to_string() != TARGET_CRS:
        man = man.to_crs(TARGET_CRS)

    roads.to_file(ROADS_2263, driver="GeoJSON")
    man.to_file(MAN_2263, driver="GeoJSON")

    # Clip to Manhattan
    man = gpd.read_file(MAN_2263)
    roads = gpd.read_file(ROADS_2263)
    man_union = make_valid(union_all(list(man.geometry)))
    roads = gpd.clip(roads, gpd.GeoDataFrame(geometry=[man_union], crs=man.crs))
    roads = roads[~roads.geometry.is_empty].copy()
    roads = roads.explode(index_parts=False, ignore_index=True)
    roads = roads[roads.geom_type == "LineString"].copy()
    roads.to_file(ROADS_CLIP, driver="GeoJSON")

def load_graph(graph_txt):
    with open(graph_txt, "r") as f:
        V, E = map(int, f.readline().split())
        list_nodes = list(map(int, f.readline().split()))
        efrom, eto, etime = [], [], []
        for line in f:
            u, v, w = line.split()
            efrom.append(int(u)); eto.append(int(v)); etime.append(float(w))
    return V, np.array(list_nodes,int), np.array(efrom,int), np.array(eto,int), np.array(etime,float)

def build_graph_from_roads():
    roads = gpd.read_file(ROADS_CLIP)
    ft_per_sec = CONST_MPH * MPH_TO_FTPS
    min_per_ft = (1.0/ft_per_sec) * SEC_TO_MIN

    rep_node = {}
    list_nodes = []
    coord_map = {}
    V = 0
    edges = []

    need_cols = {"NodeIDFrom","NodeIDTo"}
    if not need_cols.issubset(roads.columns):
        raise RuntimeError(f"roads missing {need_cols}")

    for _, r in roads.iterrows():
        u_id = int(r["NodeIDFrom"])
        v_id = int(r["NodeIDTo"])
        p0 = r.geometry.coords[0]
        p1 = r.geometry.coords[-1]
        for nid, pt in ((u_id,p0),(v_id,p1)):
            if nid not in rep_node:
                rep_node[nid] = V
                list_nodes.append(nid)
                coord_map[V] = np.array(pt, float)
                V += 1
        u = rep_node[u_id]; v = rep_node[v_id]
        length_ft = float(r.geometry.length)
        time_min  = length_ft * min_per_ft
        traf = (r.get("TrafDir") or "T").strip().upper()
        if traf in ("W","T"): edges.append((u,v,time_min))
        if traf in ("A","T"): edges.append((v,u,time_min))

    with open(GRAPH_TXT, "w") as f:
        f.write(f"{V} {len(edges)}\n")
        f.write(" ".join(map(str, list_nodes)) + "\n")
        for u,v,w in edges:
            f.write(f"{u} {v} {w}\n")

    node_xy = np.vstack([coord_map[i] for i in range(V)])
    nodes_gdf = gpd.GeoDataFrame(
        {"NodeID": list_nodes, "index": list(range(V))},
        geometry=gpd.points_from_xy(node_xy[:,0], node_xy[:,1]),
        crs=roads.crs
    )
    nodes_gdf.to_file(NODES_GDF, driver="GeoJSON")

def solve_node_potential():
    """
    Compute node potentials using shortest-path distance from destination.

    This creates a potential field with:
    - Source at start only (no sink at destination)
    - phi = distance to destination (high at source, zero at destination)
    - Greedy descent on phi leads toward destination
    """
    nodes = gpd.read_file(NODES_GDF).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]

    with open(GRAPH_TXT, "r") as f:
        V, E = map(int, f.readline().split())
        list_nodes = list(map(int, f.readline().split()))
        efrom, eto, etime = [], [], []
        for line in f:
            u,v,w = line.split()
            efrom.append(int(u)); eto.append(int(v)); etime.append(float(w))
    efrom = np.array(efrom,int); eto = np.array(eto,int); etime = np.array(etime,float)
    assert V == coords.shape[0], "Node count mismatch."

    # Build adjacency matrix for Dijkstra (use minimum time for duplicate edges)
    edge_weights = {}
    for u, v, t in zip(efrom, eto, etime):
        key = (int(u), int(v))
        if key not in edge_weights or t < edge_weights[key]:
            edge_weights[key] = t

    rows = np.array([k[0] for k in edge_weights.keys()], dtype=int)
    cols = np.array([k[1] for k in edge_weights.keys()], dtype=int)
    data = np.array(list(edge_weights.values()), dtype=float)
    adj = csr_matrix((data, (rows, cols)), shape=(V, V))

    node_index = {nid:i for i,nid in enumerate(list_nodes)}
    t = node_index[END_NODEID]  # destination node

    # Compute shortest path distances FROM destination to all nodes
    # We need reverse graph since we want distance TO destination
    # Transpose adjacency to get distances FROM destination
    adj_T = adj.T.tocsr()
    dist_from_dst = csgraph.dijkstra(adj_T, directed=True, indices=t)

    # phi = distance to destination (source has high phi, destination has phi=0)
    phi = dist_from_dst

    # Handle unreachable nodes (infinite distance)
    phi[~np.isfinite(phi)] = np.nanmax(phi[np.isfinite(phi)]) if np.any(np.isfinite(phi)) else 0.0

    nodes["phi"] = phi
    nodes.to_file(NODES_PHI, driver="GeoJSON")
    print(f"Potential field: phi ranges from {phi.min():.2f} to {phi.max():.2f}")

def compute_routes_and_plot():
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    phi    = nodes["phi"].to_numpy(float)

    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)
    assert V == coords.shape[0] == phi.shape[0], "Node count mismatch."

    idx = {nid:i for i,nid in enumerate(list_nodes)}
    src = idx[START_NODEID]; dst = idx[END_NODEID]
    # Build time map with minimum weight for duplicate edges
    tmap = {}
    for u, v, w in zip(efrom, eto, etime):
        key = (int(u), int(v))
        if key not in tmap or w < tmap[key]:
            tmap[key] = w

    # Use FORWARD greedy gradient descent from source (follows decreasing phi toward destination)
    greedy_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)
    greedy_time = path_time(greedy_nodes, tmap)

    # write routes
    def to_line(path): return LineString([coords[i] for i in path]) if len(path)>=2 else None
    out = gpd.GeoDataFrame(
        {"name": ["greedy","dijkstra"],
         "time_min":[float(greedy_time), float(dijk_time)],
         "start_id":[START_NODEID]*2, "end_id":[END_NODEID]*2},
        geometry=[to_line(greedy_nodes), to_line(dijk_nodes)],
        crs=gpd.read_file(MAN_2263).crs
    ).dropna(subset=["geometry"])
    out.to_file(ROUTES_GEOJSON, driver="GeoJSON")

    # viz with roads + legend: Dijkstra and Greedy
    _, ax = plt.subplots(figsize=(8.2, 10.5))
    try:
        man = gpd.read_file(MAN_2263)
        gpd.GeoDataFrame(geometry=[make_valid(union_all(list(man.geometry)))], crs=man.crs).boundary.plot(
            ax=ax, color="black", linewidth=1.0
        )
    except Exception:
        pass
    try:
        roads = gpd.read_file(ROADS_CLIP)
        roads.plot(ax=ax, color="lightgray", linewidth=0.35)
    except Exception:
        pass

    if len(dijk_nodes)>=2:
        C = coords[np.array(dijk_nodes)]
        ax.plot(C[:,0], C[:,1], '-', lw=3, color='magenta', label=f"Dijkstra ({dijk_time:.1f} min)")
    if len(greedy_nodes)>=2:
        C = coords[np.array(greedy_nodes)]
        ax.plot(C[:,0], C[:,1], '-', lw=3, color='cyan', label=f"Greedy ({greedy_time:.1f} min)")

    ax.plot(coords[src,0], coords[src,1], 'go', ms=8, label="Start")
    ax.plot(coords[dst,0], coords[dst,1], 'ro', ms=8, label="End")
    ax.set_aspect('equal'); ax.set_xlabel("x (ft, EPSG:2263)"); ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title("Manhattan: Greedy vs Dijkstra (25 mph edges)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(ROUTES_PNG, dpi=220)
    print(f"Greedy time: {greedy_time:.2f} min  |  Dijkstra time: {dijk_time:.2f} min")
    print("Saved:", ROUTES_PNG)

    # Greedy only PNG
    _, ax2 = plt.subplots(figsize=(8.2, 10.5))
    try:
        man = gpd.read_file(MAN_2263)
        gpd.GeoDataFrame(geometry=[make_valid(union_all(list(man.geometry)))], crs=man.crs).boundary.plot(
            ax=ax2, color="black", linewidth=1.0
        )
    except Exception:
        pass
    try:
        roads = gpd.read_file(ROADS_CLIP)
        roads.plot(ax=ax2, color="lightgray", linewidth=0.35)
    except Exception:
        pass
    if len(greedy_nodes)>=2:
        C = coords[np.array(greedy_nodes)]
        ax2.plot(C[:,0], C[:,1], '-', lw=3, color='cyan', label=f"Greedy ({greedy_time:.1f} min)")
    ax2.plot(coords[src,0], coords[src,1], 'go', ms=8, label="Start")
    ax2.plot(coords[dst,0], coords[dst,1], 'ro', ms=8, label="End")
    ax2.set_aspect('equal'); ax2.set_xlabel("x (ft, EPSG:2263)"); ax2.set_ylabel("y (ft, EPSG:2263)")
    ax2.set_title("Manhattan: Greedy route (source only, no sink)")
    ax2.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(ROUTES_SINGUM_PNG, dpi=220)
    print("Saved:", ROUTES_SINGUM_PNG)

# ----------------------- VORONOI -> K TENSORS -----------------------
def build_K_from_voronoi():
    if not os.path.exists(VORO_IN):
        raise RuntimeError(f"Missing {VORO_IN}")
    vor = gpd.read_file(VORO_IN)[["NodeID","geometry"]].copy()
    # Ensure numeric NodeID
    vor["NodeID"] = pd.to_numeric(vor["NodeID"], errors="coerce").astype("Int64")

    man = gpd.read_file(MAN_2263)
    man_union = make_valid(union_all(list(man.geometry)))
    vor = gpd.overlay(vor, gpd.GeoDataFrame(geometry=[man_union], crs=man.crs),
                      how="intersection", keep_geom_type=True)
    vor = vor.dissolve(by="NodeID", as_index=False)
    vor["area_ft2"] = vor.geometry.area

    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)
    nodes = gpd.read_file(NODES_GDF).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    assert coords.shape[0] == V == len(list_nodes), "Node mismatch."

    # =========================================================================
    # Compute effective conductivity tensor k_{ij} = k^1_{ij} + k^2_{ij}
    # Following Paper Equations 20-21 for non-central-symmetric networks
    # =========================================================================

    # Build adjacency: for each node, collect all connected links with their properties
    # Each link contributes: conductance G = 1/time, direction n, half-length l_p = L/2
    node_links = defaultdict(list)  # node -> list of (neighbor, G, L, n_vec)

    for u, v, t in zip(efrom, eto, etime):
        if t > 0:
            G = 1.0 / t
            pa, pb = coords[u], coords[v]
            d = pb - pa
            L = np.linalg.norm(d)
            if L > 0:
                n = d / L  # unit direction from u to v
                node_links[u].append((v, G, L, n))
                node_links[v].append((u, G, L, -n))  # reverse direction for other endpoint

    # Initialize k^1 and k^2 components
    K1_xx = np.zeros(V); K1_xy = np.zeros(V); K1_yy = np.zeros(V)
    K2_xx = np.zeros(V); K2_xy = np.zeros(V); K2_yy = np.zeros(V)

    for node in range(V):
        links = node_links[node]
        if not links:
            continue

        # =====================================================================
        # Paper Eq. 20: k^1_{ij} = Σ K^I A x^I_i x^I_j / (V_s l_p^0)
        #
        # With x^I = l_p^0 * n^I (half-link vector to cutting point)
        # and K^I * A / l_p^0 ∝ G (conductance = 1/time)
        #
        # k^1_{ij} = Σ (K^I A l_p^0) * n^I_i * n^I_j / V_s
        #          = Σ (G * L² / 2) * n_i * n_j / V_s
        # =====================================================================
        for (neighbor, G, L, n) in links:
            l_p = L / 2.0  # half-link length
            # Factor = K^I * A * l_p^0 ≈ G * L * l_p = G * L² / 2
            factor = G * L * l_p  # = G * L² / 2
            K1_xx[node] += factor * n[0] * n[0]
            K1_xy[node] += factor * n[0] * n[1]
            K1_yy[node] += factor * n[1] * n[1]

        # =====================================================================
        # Paper Eq. 21: k^2_{ij} (correction for non-central-symmetric networks)
        #
        # k^2_{ij} = -[Σ K^I A n^I_i] * [Σ K^I A n^I_j] / [Σ K^I A / l_p^0 * V_s]
        #
        # Define: S_i = Σ (K^I A) * n^I_i ≈ Σ G * L * n_i
        #         D   = Σ (K^I A / l_p^0) ≈ Σ 2 * G
        #
        # Then: k^2_{ij} = -S_i * S_j / (D * V_s)
        # =====================================================================
        S_x = 0.0
        S_y = 0.0
        D = 0.0

        for (neighbor, G, L, n) in links:
            # S_i = Σ (K^I A) * n_i, where K^I A ∝ G * L (conductance × length)
            KA = G * L  # proportional to K^I * A
            S_x += KA * n[0]
            S_y += KA * n[1]
            # D = Σ K^I A / l_p^0 = Σ 2 * G
            D += 2.0 * G

        # k^2_{ij} = -S_i * S_j / (D * V_s)
        # Note: We'll divide by V_s (area) later along with k^1
        if D > 0:
            K2_xx[node] = -S_x * S_x / D
            K2_xy[node] = -S_x * S_y / D
            K2_yy[node] = -S_y * S_y / D

    # Get singum areas (V_s) from Voronoi cells
    nodeid_to_idx = {nid: i for i, nid in enumerate(list_nodes)}
    idx_to_nodeid = np.array(list_nodes, int)
    area = np.zeros(V)
    vor_keep = vor[vor["NodeID"].isin(nodeid_to_idx)].copy()
    area_idx = vor_keep["NodeID"].map(nodeid_to_idx).to_numpy(int)
    area[area_idx] = vor_keep["area_ft2"].to_numpy(float)
    safe = np.where(area > 0, area, np.nan)

    # =========================================================================
    # Final conductivity: k_{ij} = (k^1_{ij} + k^2_{ij}) / V_s
    # =========================================================================
    Kxx = (K1_xx + K2_xx) / safe
    Kxy = (K1_xy + K2_xy) / safe
    Kyy = (K1_yy + K2_yy) / safe
    Ktab = pd.DataFrame({"NodeID": idx_to_nodeid, "Kxx":Kxx, "Kxy":Kxy, "Kyy":Kyy})
    vor = vor.merge(Ktab, on="NodeID", how="left")
    vor.to_file(VORO_OUT, driver="GeoJSON")

    cent = vor.geometry.centroid
    np.savetxt(
        K_CENTROID_CSV,
        np.c_[cent.x.values, cent.y.values,
              vor["Kxx"].to_numpy(float),
              vor["Kxy"].to_numpy(float),
              vor["Kyy"].to_numpy(float)],
        delimiter=",", header="x_ft,y_ft,Kxx,Kxy,Kyy", comments=""
    )
    print("Wrote:", VORO_OUT, "and", K_CENTROID_CSV)

    # Plot Voronoi cells/edges
    fig, ax = plt.subplots(figsize=(8.2, 10.5))
    man = gpd.read_file(MAN_2263)
    gpd.GeoDataFrame(geometry=[make_valid(union_all(list(man.geometry)))], crs=man.crs).boundary.plot(
        ax=ax, color="black", linewidth=1.0
    )
    vor.boundary.plot(ax=ax, color="steelblue", linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_xlabel("x (ft, EPSG:2263)"); ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title("Voronoi cells (singum)")
    plt.tight_layout(); plt.savefig(VORO_PNG, dpi=220)
    print("Saved:", VORO_PNG)

# ----------------------- CONTINUUM HEATMAP (ROBUST) -----------------------
def _manhattan_mask_and_grid(nx=GRID_NX, ny=GRID_NY, extra_buffer_ft=GRID_BUFFER_FT):
    man = gpd.read_file(MAN_2263)
    man_union = make_valid(union_all(list(man.geometry)))
    minx, miny, maxx, maxy = man_union.bounds
    minx -= extra_buffer_ft; miny -= extra_buffer_ft
    maxx += extra_buffer_ft; maxy += extra_buffer_ft
    X, Y = np.meshgrid(np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny))

    try:
        from matplotlib.path import Path
        geoms = list(man_union.geoms) if man_union.geom_type=="MultiPolygon" else [man_union]
        P = np.c_[X.ravel(), Y.ravel()]
        inside = np.zeros(P.shape[0], dtype=bool)
        for poly in geoms:
            ex = np.c_[*poly.exterior.xy].astype(float)
            inside |= Path(ex).contains_points(P)
            for ring in poly.interiors:
                ring_path = Path(np.c_[*ring.xy].astype(float))
                hole = ring_path.contains_points(P)
                inside[hole] = False
        inside = inside.reshape(Y.shape)
    except Exception:
        pts = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Y.ravel()), crs=man.crs)
        inside = pts.within(man_union).to_numpy().reshape(Y.shape)

    return man_union, X, Y, inside

def _interp_K_no_nans(vor_gdf, X, Y, inside, blur_sigma=0.8):
    C = vor_gdf.geometry.centroid
    P = np.c_[C.x.values, C.y.values]
    Kxxv = vor_gdf["Kxx"].to_numpy(float)
    Kxyv = vor_gdf["Kxy"].to_numpy(float)
    Kyyv = vor_gdf["Kyy"].to_numpy(float)

    fxx = LinearNDInterpolator(P, Kxxv, fill_value=np.nan)
    fxy = LinearNDInterpolator(P, Kxyv, fill_value=np.nan)
    fyy = LinearNDInterpolator(P, Kyyv, fill_value=np.nan)
    Kxxg = fxx(X, Y); Kxyg = fxy(X, Y); Kyyg = fyy(X, Y)

    for Z, Vv in ((Kxxg, Kxxv),(Kxyg, Kxyv),(Kyyg, Kyyv)):
        nanmask = inside & (~np.isfinite(Z))
        if nanmask.any():
            Z[nanmask] = griddata(P, Vv, (X[nanmask], Y[nanmask]), method="nearest")
        Z[~inside] = np.nan

    return (nan_gaussian_blur(Kxxg, blur_sigma),
            nan_gaussian_blur(Kxyg, blur_sigma),
            nan_gaussian_blur(Kyyg, blur_sigma))

def solve_continuum_and_plot():
    # 0) Voronoi with K (build if needed)
    try:
        vor = gpd.read_file(VORO_OUT)
        if not {"Kxx","Kxy","Kyy"}.issubset(vor.columns):
            raise RuntimeError
    except Exception:
        print("Rebuilding K from Voronoi…")
        build_K_from_voronoi()
        vor = gpd.read_file(VORO_OUT)

    # 1) Grid + island mask
    man_union, X, Y, inside = _manhattan_mask_and_grid()
    Kxxg, Kxyg, Kyyg = _interp_K_no_nans(vor, X, Y, inside, blur_sigma=0.8)

    # One-pixel shoreline ring
    sh = np.zeros_like(inside, dtype=bool)
    sh |= inside & np.roll(~inside,  1, axis=0)
    sh |= inside & np.roll(~inside, -1, axis=0)
    sh |= inside & np.roll(~inside,  1, axis=1)
    sh |= inside & np.roll(~inside, -1, axis=1)
    shoreline = sh

    # 2) Always create two Dirichlet shoreline arcs: SOUTH=1, NORTH=0
    sy, sx = np.where(shoreline)
    if sy.size == 0:
        D1_mask = np.zeros_like(inside, bool); D0_mask = np.zeros_like(inside, bool)
        D1_mask[0,0]   = True
        D0_mask[-1,-1] = True
    else:
        yvals  = Y[sy, sx]
        y_lo   = np.percentile(yvals, 10)  # southern 10% of coastline
        y_hi   = np.percentile(yvals, 90)  # northern 10%
        D1_mask = shoreline & (Y <= y_lo)  # φ=1
        D0_mask = shoreline & (Y >= y_hi)  # φ=0

    # 3) Solve using FEM (P1 triangular elements with anisotropic K)
    # Dirichlet BCs: south shoreline = 1, north shoreline = 0
    V = solve_fem_anisotropic(
        X, Y, inside, Kxxg, Kxyg, Kyyg,
        dirichlet_masks=[D1_mask, D0_mask],
        dirichlet_values=[1.0, 0.0]
    )

    # 5) Plot: true heatmap + contours + routes
    V_plot = V.copy()
    V_plot[~inside] = np.nan
    # stretch to [0,1] for a vivid colormap
    vmin = np.nanmin(V_plot); vmax = np.nanmax(V_plot)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        V_plot = (V_plot - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(8.8, 11.2))
    gpd.GeoDataFrame(geometry=[man_union], crs=gpd.read_file(MAN_2263).crs)\
      .boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=1)

    im = ax.pcolormesh(X, Y, V_plot, shading="auto", cmap="viridis", zorder=2)
    ax.contour(X, Y, V_plot, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)

    # roads & routes on top
    try:
        roads = gpd.read_file(ROADS_CLIP)
        roads.plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
    except Exception:
        pass
    try:
        routes = gpd.read_file(ROUTES_GEOJSON)
        for _, r in routes.iterrows():
            col = 'cyan' if r["name"]=="singum" else 'magenta'
            ax.plot(*r.geometry.xy, '-', lw=3, color=col, zorder=5)
    except Exception:
        pass

    ax.set_aspect('equal')
    ax.set_xlabel("x (ft, EPSG:2263)"); ax.set_ylabel("y (ft, EPSG:2263)")
    plt.colorbar(im, ax=ax, label="Potential φ (scaled)")
    ax.set_title("Continuum potential on Manhattan (robust K)")
    plt.tight_layout(); plt.savefig(HEATMAP_PNG, dpi=220)
    print("Saved:", HEATMAP_PNG)

# ----------------------- FIG 5C: NBA GAME OVER -----------------------
def case_5c_nba_game(save_prefix="output/fig5c_nba", show=False):
    """
    Fig 5c: NBA game over at Madison Square Garden.
    20,000 people leave MSG (source) and travel to various destinations.

    Uses graph Laplacian with edge conductances (undirected):
    - MSG nodes have phi=1 (source)
    - Destination nodes have phi=0 (sink)
    - Greedy routing follows decreasing phi

    Shows potential distribution and compares Greedy vs Dijkstra
    paths to several destinations.
    """
    print("\n" + "="*60)
    print("Fig 5c: NBA Game Over - MSG dispersal scenario")
    print("="*60)

    # Load graph and nodes
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    N_nodes, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)

    # Build undirected time map
    tmap_undir = {}
    for u, v, w in zip(efrom, eto, etime):
        key1 = (int(u), int(v))
        key2 = (int(v), int(u))
        if key1 not in tmap_undir or w < tmap_undir[key1]:
            tmap_undir[key1] = w
        if key2 not in tmap_undir or w < tmap_undir[key2]:
            tmap_undir[key2] = w

    # Get MSG location in EPSG:2263
    msg_pt = lonlat_to_2263(MSG_LONLAT)
    msg_xy = np.array([msg_pt.x, msg_pt.y])

    # Get Manhattan boundary and create inside mask
    man = gpd.read_file(MAN_2263)
    man_union = make_valid(union_all(list(man.geometry)))

    # Find MSG source nodes (phi=1)
    dists_to_msg = np.linalg.norm(coords - msg_xy, axis=1)
    source_nodes = set(np.where(dists_to_msg <= MSG_RADIUS_FT)[0])
    if len(source_nodes) == 0:
        source_nodes = {nearest_node(coords, msg_xy)}

    print(f"MSG source nodes: {len(source_nodes)}")

    # Grid setup for plotting
    man_bounds = man_union.bounds
    xg = np.linspace(man_bounds[0] - GRID_BUFFER_FT, man_bounds[2] + GRID_BUFFER_FT, GRID_NX)
    yg = np.linspace(man_bounds[1] - GRID_BUFFER_FT, man_bounds[3] + GRID_BUFFER_FT, GRID_NY)
    X, Y = np.meshgrid(xg, yg)

    # Create Manhattan inside mask
    try:
        from matplotlib.path import Path
        geoms = list(man_union.geoms) if man_union.geom_type == "MultiPolygon" else [man_union]
        P = np.c_[X.ravel(), Y.ravel()]
        inside = np.zeros(P.shape[0], dtype=bool)
        for poly in geoms:
            ex = np.c_[*poly.exterior.xy].astype(float)
            inside |= Path(ex).contains_points(P)
            for ring in poly.interiors:
                ring_path = Path(np.c_[*ring.xy].astype(float))
                inside[ring_path.contains_points(P)] = False
        inside = inside.reshape(Y.shape)
    except Exception:
        pts = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Y.ravel()), crs=man.crs)
        inside = pts.within(man_union).to_numpy().reshape(Y.shape)

    # Get MSG source node (center)
    src = nearest_node(coords, msg_xy)

    # Build undirected edge conductances
    edge_conductance = defaultdict(float)
    for u, v, t in zip(efrom, eto, etime):
        if t > 0:
            G = 1.0 / t
            a, b = min(u, v), max(u, v)
            edge_conductance[(a, b)] += G

    # Build node neighbors from undirected conductances
    node_neighbors = defaultdict(list)
    for (a, b), G in edge_conductance.items():
        node_neighbors[a].append((b, G))
        node_neighbors[b].append((a, G))

    # Build undirected edge arrays
    efrom_undir = np.concatenate([efrom, eto])
    eto_undir = np.concatenate([eto, efrom])
    etime_undir = np.concatenate([etime, etime])

    # Compute paths to each destination using graph Laplacian
    results_table = []
    destination_paths = {}
    DEST_RADIUS_FT = 400.0  # radius for destination sink region

    for dest_name, dest_lonlat in DESTINATIONS.items():
        dest_pt = lonlat_to_2263(dest_lonlat)
        dest_xy = np.array([dest_pt.x, dest_pt.y])
        dst = nearest_node(coords, dest_xy)

        # Find sink nodes near destination (phi=0)
        dists_to_dest = np.linalg.norm(coords - dest_xy, axis=1)
        sink_nodes = set(np.where(dists_to_dest <= DEST_RADIUS_FT)[0])
        if len(sink_nodes) == 0:
            sink_nodes = {dst}

        # Solve graph Laplacian: MSG=source(phi=1), dest=sink(phi=0)
        L_rows, L_cols, L_data = [], [], []
        b_graph = np.zeros(N_nodes, float)

        for node in range(N_nodes):
            if node in source_nodes:
                # Dirichlet BC: phi = 1 at MSG (source)
                L_rows.append(node); L_cols.append(node); L_data.append(1.0)
                b_graph[node] = 1.0
            elif node in sink_nodes:
                # Dirichlet BC: phi = 0 at destination (sink)
                L_rows.append(node); L_cols.append(node); L_data.append(1.0)
                b_graph[node] = 0.0
            else:
                # Interior node
                diag = 0.0
                for neighbor, G in node_neighbors[node]:
                    diag += G
                    L_rows.append(node); L_cols.append(neighbor); L_data.append(-G)
                if diag > 0:
                    L_rows.append(node); L_cols.append(node); L_data.append(diag)
                else:
                    L_rows.append(node); L_cols.append(node); L_data.append(1.0)
                    b_graph[node] = 0.5

        L_graph = csr_matrix((np.array(L_data), (np.array(L_rows), np.array(L_cols))), shape=(N_nodes, N_nodes))
        L_graph = L_graph + csr_matrix((np.full(N_nodes, 1e-10), (np.arange(N_nodes), np.arange(N_nodes))), shape=(N_nodes, N_nodes))

        phi_nodes = spsolve(L_graph, b_graph)

        # Dijkstra path (on undirected graph)
        dijk_nodes, dijk_time = dijkstra_path(coords, efrom_undir, eto_undir, etime_undir, src, dst)

        # Greedy: follow decreasing phi on undirected graph (pass sink_nodes to stop early)
        greedy_nodes = greedy_path_on_nodes(phi_nodes, efrom_undir, eto_undir, src, dst, coords=coords, sink_nodes=sink_nodes)

        # Check if greedy reached destination or any sink node
        greedy_reached = len(greedy_nodes) > 0 and (greedy_nodes[-1] == dst or greedy_nodes[-1] in sink_nodes)
        greedy_time = path_time(greedy_nodes, tmap_undir)

        if not greedy_reached:
            greedy_time_str = f"{greedy_time:.2f} (INCOMPLETE)" if np.isfinite(greedy_time) else "N/A"
            greedy_diff_str = "N/A"
        else:
            greedy_time_str = f"{greedy_time:.2f}" if np.isfinite(greedy_time) else "N/A"
            greedy_diff_str = f"{100*(greedy_time - dijk_time)/dijk_time:.1f}" if np.isfinite(greedy_time) and dijk_time > 0 else "N/A"

        results_table.append({
            'Destination': dest_name,
            'Dijkstra (min)': f"{dijk_time:.2f}",
            'Greedy (min)': greedy_time_str,
            'Greedy Diff (%)': greedy_diff_str
        })

        destination_paths[dest_name] = {
            'dijkstra': dijk_nodes,
            'greedy_path': greedy_nodes,
            'dijk_time': dijk_time,
            'greedy_time': greedy_time,
            'dest_xy': dest_xy,
            'dst_node': dst,
            'phi_nodes': phi_nodes,
            'sink_nodes': sink_nodes
        }

        print(f"  {dest_name}: Dijkstra={dijk_time:.2f}, Greedy={greedy_time_str} min")

    # Print results table
    print("\n" + "-"*70)
    print("Travel Time Comparison Table:")
    print("-"*70)
    print(f"{'Destination':<15} {'Dijkstra':<12} {'Greedy':<12} {'Diff (%)':<12}")
    print("-"*70)
    for row in results_table:
        print(f"{row['Destination']:<15} {row['Dijkstra (min)']:<12} {row['Greedy (min)']:<12} {row['Greedy Diff (%)']:<12}")
    print("-"*70)

    # ===== PLOT: Separate graph for each destination =====
    colors_list = ['cyan', 'magenta', 'red', 'yellow', 'lime']

    for i, (dest_name, pdata) in enumerate(destination_paths.items()):
        color = colors_list[i % len(colors_list)]

        # Interpolate phi to grid for heatmap visualization
        phi_nodes = pdata['phi_nodes']
        Phi_lin = griddata(coords, phi_nodes, (X, Y), method='linear')
        Phi_nn = griddata(coords, phi_nodes, (X, Y), method='nearest')
        Phi = np.where(np.isfinite(Phi_lin), Phi_lin, Phi_nn)

        Phi_plot = Phi.copy()
        Phi_plot[~inside] = np.nan

        # Normalize to [0,1]
        vmin, vmax = np.nanmin(Phi_plot), np.nanmax(Phi_plot)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            Phi_plot = (Phi_plot - vmin) / (vmax - vmin)

        # Create figure for this destination
        fig, ax = plt.subplots(figsize=(8.8, 11.2))

        # Manhattan boundary
        gpd.GeoDataFrame(geometry=[man_union], crs=man.crs).boundary.plot(
            ax=ax, color="black", linewidth=1.0, zorder=1)

        # Potential heatmap
        im = ax.pcolormesh(X, Y, Phi_plot, shading="auto", cmap="viridis", zorder=2)

        # Contour lines
        ax.contour(X, Y, Phi_plot, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)

        # Roads on top
        try:
            roads = gpd.read_file(ROADS_CLIP)
            roads.plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
        except Exception:
            pass

        # Dijkstra path (solid magenta)
        if pdata['dijkstra'] and len(pdata['dijkstra']) >= 2:
            C = coords[np.array(pdata['dijkstra'])]
            ax.plot(C[:, 0], C[:, 1], '-', lw=3, color='magenta',
                    label=f"Dijkstra ({pdata['dijk_time']:.1f} min)", zorder=5)

        # Greedy path (solid cyan)
        if pdata['greedy_path'] and len(pdata['greedy_path']) >= 2:
            C = coords[np.array(pdata['greedy_path'])]
            ax.plot(C[:, 0], C[:, 1], '-', lw=3, color='cyan',
                    label=f"Greedy ({pdata['greedy_time']:.1f} min)", zorder=6)

        # Destination marker
        ax.plot(pdata['dest_xy'][0], pdata['dest_xy'][1], 'o', ms=12, color=color,
                markeredgecolor='black', markeredgewidth=1.5, label=dest_name, zorder=8)

        # MSG source marker
        ax.plot(msg_xy[0], msg_xy[1], 'g*', ms=18, markeredgecolor='black',
                markeredgewidth=1, label='MSG (Source)', zorder=10)

        ax.set_aspect('equal')
        ax.set_xlabel("x (ft, EPSG:2263)")
        ax.set_ylabel("y (ft, EPSG:2263)")
        ax.set_title(f"Fig 5c: MSG to {dest_name}\n(Graph Laplacian, Greedy vs Dijkstra)")
        plt.colorbar(im, ax=ax, label="Potential φ (scaled)")
        ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        dest_png_path = f"{save_prefix}_{dest_name.lower().replace(' ', '_')}.png"
        plt.savefig(dest_png_path, dpi=220)
        print(f"Saved: {dest_png_path}")

        if show:
            plt.show()
        else:
            plt.close()

    return results_table, destination_paths


# ----------------------- FIG 5C REVERSE: NBA GAME ARRIVAL -----------------------
def case_5c_nba_arrival(save_prefix="output/fig5c_nba_arrival", show=False):
    """
    Fig 5c (Reverse): NBA game at Madison Square Garden - ARRIVAL scenario.
    20,000 people travel FROM various locations across NYC TO MSG (sink).

    This is the reverse of case_5c_nba_game:
    - MSG is the SINK (Dirichlet BC φ=0)
    - Distributed sources across Manhattan (Neumann flux from boundaries)
    - Solves continuum PDE: ∇·(K∇φ) = 0 with anisotropic K tensor
    - Greedy routing follows decreasing phi toward MSG

    Shows potential distribution and compares Greedy vs Dijkstra
    paths from several starting locations to MSG.
    """
    print("\n" + "="*60)
    print("Fig 5c (Reverse): NBA Game Arrival - MSG convergence scenario")
    print("="*60)

    # ========== 0) Load Voronoi K tensor ==========
    try:
        vor = gpd.read_file(VORO_OUT)
        if not {"Kxx","Kxy","Kyy"}.issubset(vor.columns):
            raise RuntimeError
    except Exception:
        print("Rebuilding K from Voronoi...")
        build_K_from_voronoi()
        vor = gpd.read_file(VORO_OUT)

    # Load graph and nodes
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    N_nodes, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)

    # Build time map with minimum weight for duplicate edges
    tmap = {}
    for u, v, w in zip(efrom, eto, etime):
        key = (int(u), int(v))
        if key not in tmap or w < tmap[key]:
            tmap[key] = w

    # ========== 1) Grid + island mask ==========
    man_union, X, Y, inside = _manhattan_mask_and_grid()
    man = gpd.read_file(MAN_2263)
    Kxxg, Kxyg, Kyyg = _interp_K_no_nans(vor, X, Y, inside, blur_sigma=0.8)

    # Grid metrics
    nxg = X.shape[1]; nyg = X.shape[0]

    # Get MSG location in EPSG:2263 - this is the SINK
    msg_pt_local = gpd.GeoSeries([Point(MSG_LONLAT)], crs="EPSG:4326").to_crs(man.crs).iloc[0]
    msg_xy = np.array([msg_pt_local.x, msg_pt_local.y])

    # Find MSG node (sink)
    dst = nearest_node(coords, msg_xy)

    print(f"MSG sink at node {dst}")
    print(f"Grid size: {nxg} x {nyg}")

    # ========== 2) Create MSG sink region (Dirichlet BC φ=0) ==========
    msg_circle = msg_pt_local.buffer(MSG_RADIUS_FT)
    from shapely.prepared import prep
    msg_ring = prep(msg_circle)

    # Create mask for MSG sink region
    MSG_mask = np.zeros_like(inside, dtype=bool)
    xs = X[0,:]; ys = Y[:,0]
    for j in range(nyg):
        for i in range(nxg):
            if inside[j, i] and msg_ring.contains(Point(xs[i], ys[j])):
                MSG_mask[j, i] = True

    # ========== 3) Create source regions (Dirichlet BC φ=1) at distributed locations ==========
    SOURCES = {
        "Grand Central": GRAND_CENTRAL_LONLAT,
        "Columbia": COLUMBIA_LONLAT,
        "NYU": NYU_LONLAT,
        "Times Square": TIMES_SQUARE_LONLAT,
        "Wall Street": WALL_STREET_LONLAT,
    }

    SOURCE_RADIUS_FT = 400.0  # radius for source regions
    source_masks = {}
    source_xy_dict = {}

    for src_name, src_lonlat in SOURCES.items():
        src_pt = gpd.GeoSeries([Point(src_lonlat)], crs="EPSG:4326").to_crs(man.crs).iloc[0]
        source_xy_dict[src_name] = np.array([src_pt.x, src_pt.y])
        src_circle = src_pt.buffer(SOURCE_RADIUS_FT)
        src_ring = prep(src_circle)

        src_mask = np.zeros_like(inside, dtype=bool)
        for j in range(nyg):
            for i in range(nxg):
                if inside[j, i] and src_ring.contains(Point(xs[i], ys[j])):
                    src_mask[j, i] = True
        source_masks[src_name] = src_mask

    # Combined source mask (all sources have φ=1)
    all_sources_mask = np.zeros_like(inside, dtype=bool)
    for mask in source_masks.values():
        all_sources_mask |= mask

    # ========== 4) Solve using FEM (P1 triangular elements with anisotropic K) ==========
    # Dirichlet BCs: MSG sink = 0, source regions = 1
    print("Solving continuum PDE using FEM for visualization...")
    V = solve_fem_anisotropic(
        X, Y, inside, Kxxg, Kxyg, Kyyg,
        dirichlet_masks=[MSG_mask, all_sources_mask],
        dirichlet_values=[0.0, 1.0]
    )
    print("FEM solve complete.")

    # ========== 5) Solve graph Laplacian for routing ==========
    # Treat all edges as UNDIRECTED (two-way) for the Laplacian.
    # This ensures phi decreases properly along graph edges.
    print("Solving graph Laplacian (undirected edges) for routing...")

    # Build undirected edge conductances: G(u,v) = 1/time, symmetric
    edge_conductance = defaultdict(float)
    for u, v, t in zip(efrom, eto, etime):
        if t > 0:
            G = 1.0 / t
            # Make undirected: add conductance to both (u,v) and (v,u)
            a, b = min(u, v), max(u, v)
            edge_conductance[(a, b)] += G

    # Find sink nodes (near MSG) - Dirichlet BC phi=0
    dists_to_msg = np.linalg.norm(coords - msg_xy, axis=1)
    sink_nodes = set(np.where(dists_to_msg <= MSG_RADIUS_FT)[0])
    if len(sink_nodes) == 0:
        sink_nodes = {dst}

    # Find source nodes (near each source location) - Dirichlet BC phi=1
    source_nodes = set()
    for src_xy_loc in source_xy_dict.values():
        dists_to_src = np.linalg.norm(coords - src_xy_loc, axis=1)
        src_near = np.where(dists_to_src <= SOURCE_RADIUS_FT)[0]
        source_nodes.update(src_near)
    if len(source_nodes) == 0:
        for src_xy_loc in source_xy_dict.values():
            source_nodes.add(nearest_node(coords, src_xy_loc))

    # Build symmetric graph Laplacian
    L_rows, L_cols, L_data = [], [], []
    b_graph = np.zeros(N_nodes, float)

    # First, build adjacency from undirected conductances
    node_neighbors = defaultdict(list)  # node -> [(neighbor, conductance), ...]
    for (a, b), G in edge_conductance.items():
        node_neighbors[a].append((b, G))
        node_neighbors[b].append((a, G))

    for node in range(N_nodes):
        if node in sink_nodes:
            # Dirichlet BC: phi = 0 at sink
            L_rows.append(node); L_cols.append(node); L_data.append(1.0)
            b_graph[node] = 0.0
        elif node in source_nodes:
            # Dirichlet BC: phi = 1 at sources
            L_rows.append(node); L_cols.append(node); L_data.append(1.0)
            b_graph[node] = 1.0
        else:
            # Interior node: sum_j G_ij * (phi_i - phi_j) = 0
            diag = 0.0
            for neighbor, G in node_neighbors[node]:
                diag += G
                L_rows.append(node); L_cols.append(neighbor); L_data.append(-G)
            if diag > 0:
                L_rows.append(node); L_cols.append(node); L_data.append(diag)
            else:
                # Isolated node - set to average
                L_rows.append(node); L_cols.append(node); L_data.append(1.0)
                b_graph[node] = 0.5

    L_graph = csr_matrix((np.array(L_data), (np.array(L_rows), np.array(L_cols))), shape=(N_nodes, N_nodes))

    # Small regularization
    L_graph = L_graph + csr_matrix((np.full(N_nodes, 1e-10), (np.arange(N_nodes), np.arange(N_nodes))), shape=(N_nodes, N_nodes))

    phi_nodes = spsolve(L_graph, b_graph)
    print(f"Graph Laplacian solved. Node phi range: {np.nanmin(phi_nodes):.4f} to {np.nanmax(phi_nodes):.4f}")

    # Build undirected edge arrays for greedy routing (treat one-way as two-way)
    efrom_undir = np.concatenate([efrom, eto])
    eto_undir = np.concatenate([eto, efrom])

    # ========== 6) Compute routes from each source to MSG ==========
    results_table = []
    source_paths = {}

    # Build undirected time map for path_time calculation
    tmap_undir = {}
    for u, v, w in zip(efrom, eto, etime):
        key1 = (int(u), int(v))
        key2 = (int(v), int(u))
        if key1 not in tmap_undir or w < tmap_undir[key1]:
            tmap_undir[key1] = w
        if key2 not in tmap_undir or w < tmap_undir[key2]:
            tmap_undir[key2] = w

    for src_name, src_lonlat in SOURCES.items():
        src_xy_local = source_xy_dict[src_name]
        src = nearest_node(coords, src_xy_local)

        # Dijkstra path (on UNDIRECTED graph for fair comparison)
        etime_undir = np.concatenate([etime, etime])
        dijk_nodes, dijk_time = dijkstra_path(coords, efrom_undir, eto_undir, etime_undir, src, dst)

        # Greedy: follow decreasing phi on UNDIRECTED graph (pass sink_nodes to stop early)
        greedy_nodes = greedy_path_on_nodes(phi_nodes, efrom_undir, eto_undir, src, dst, coords=coords, sink_nodes=sink_nodes)

        # Check if greedy path reached destination OR any sink node (all have phi=0)
        greedy_reached_dst = len(greedy_nodes) > 0 and (greedy_nodes[-1] == dst or greedy_nodes[-1] in sink_nodes)
        greedy_time = path_time(greedy_nodes, tmap_undir)

        # If greedy didn't reach destination, mark as incomplete
        if not greedy_reached_dst:
            print(f"  WARNING: Greedy path from {src_name} did not reach MSG (stopped at node {greedy_nodes[-1] if greedy_nodes else 'N/A'}, {len(greedy_nodes)} steps)")
            greedy_time_str = f"{greedy_time:.2f} (INCOMPLETE)" if np.isfinite(greedy_time) else "N/A"
            greedy_diff_str = "N/A (incomplete)"
        else:
            greedy_time_str = f"{greedy_time:.2f}" if np.isfinite(greedy_time) else "N/A"
            greedy_diff_str = f"{100*(greedy_time - dijk_time)/dijk_time:.1f}" if np.isfinite(greedy_time) and dijk_time > 0 else "N/A"

        results_table.append({
            'Source': src_name,
            'Dijkstra (min)': f"{dijk_time:.2f}",
            'Greedy (min)': greedy_time_str,
            'Greedy Diff (%)': greedy_diff_str,
            'reached_dst': greedy_reached_dst
        })

        source_paths[src_name] = {
            'dijkstra': dijk_nodes,
            'greedy_path': greedy_nodes,
            'dijk_time': dijk_time,
            'greedy_time': greedy_time,
            'src_xy': src_xy_local,
            'src_node': src,
            'reached_dst': greedy_reached_dst
        }

        print(f"  {src_name} -> MSG: Dijkstra={dijk_time:.2f}, Greedy={greedy_time_str} min")

    # Print results table
    print("\n" + "-"*70)
    print("Travel Time Comparison Table (Sources -> MSG):")
    print("-"*70)
    print(f"{'Source':<15} {'Dijkstra':<12} {'Greedy':<12} {'Diff (%)':<12}")
    print("-"*70)
    for row in results_table:
        print(f"{row['Source']:<15} {row['Dijkstra (min)']:<12} {row['Greedy (min)']:<12} {row['Greedy Diff (%)']:<12}")
    print("-"*70)

    # ========== 7) PLOT: Separate graph for each source ==========
    colors_list = ['cyan', 'magenta', 'red', 'yellow', 'lime']

    # Prepare potential for plotting
    V_plot = V.copy()
    V_plot[~inside] = np.nan
    vmin, vmax = np.nanmin(V_plot), np.nanmax(V_plot)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        V_plot = (V_plot - vmin) / (vmax - vmin)

    for i, (src_name, pdata) in enumerate(source_paths.items()):
        color = colors_list[i % len(colors_list)]

        # Create figure for this source
        fig, ax = plt.subplots(figsize=(8.8, 11.2))

        # Manhattan boundary
        gpd.GeoDataFrame(geometry=[man_union], crs=man.crs).boundary.plot(
            ax=ax, color="black", linewidth=1.0, zorder=1)

        # Potential heatmap (continuum solution)
        im = ax.pcolormesh(X, Y, V_plot, shading="auto", cmap="viridis", zorder=2)

        # Contour lines
        ax.contour(X, Y, V_plot, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)

        # Roads on top
        try:
            roads = gpd.read_file(ROADS_CLIP)
            roads.plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
        except Exception:
            pass

        # Dijkstra path (solid magenta)
        if pdata['dijkstra'] and len(pdata['dijkstra']) >= 2:
            C = coords[np.array(pdata['dijkstra'])]
            ax.plot(C[:, 0], C[:, 1], '-', lw=3, color='magenta',
                    label=f"Dijkstra ({pdata['dijk_time']:.1f} min)", zorder=5)

        # Greedy path (solid cyan)
        if pdata['greedy_path'] and len(pdata['greedy_path']) >= 2:
            C = coords[np.array(pdata['greedy_path'])]
            ax.plot(C[:, 0], C[:, 1], '-', lw=3, color='cyan',
                    label=f"Greedy ({pdata['greedy_time']:.1f} min)", zorder=6)

        # Source marker
        ax.plot(pdata['src_xy'][0], pdata['src_xy'][1], 'o', ms=12, color=color,
                markeredgecolor='black', markeredgewidth=1.5, label=f"{src_name} (Source)", zorder=8)

        # MSG sink marker
        ax.plot(msg_xy[0], msg_xy[1], 'r*', ms=18, markeredgecolor='black',
                markeredgewidth=1, label='MSG (Sink)', zorder=10)

        ax.set_aspect('equal')
        ax.set_xlabel("x (ft, EPSG:2263)")
        ax.set_ylabel("y (ft, EPSG:2263)")
        ax.set_title(f"Fig 5c (Arrival): {src_name} to MSG\n(Continuum PDE, Greedy vs Dijkstra)")
        plt.colorbar(im, ax=ax, label="Potential φ (scaled)")
        ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        src_png_path = f"{save_prefix}_{src_name.lower().replace(' ', '_')}.png"
        plt.savefig(src_png_path, dpi=220)
        print(f"Saved: {src_png_path}")

        if show:
            plt.show()
        else:
            plt.close()

    # ========== 8) COMBINED PLOT: All sources converging on MSG ==========
    fig, ax = plt.subplots(figsize=(8.8, 11.2))

    # Manhattan boundary
    gpd.GeoDataFrame(geometry=[man_union], crs=man.crs).boundary.plot(
        ax=ax, color="black", linewidth=1.0, zorder=1)

    # Potential heatmap
    im = ax.pcolormesh(X, Y, V_plot, shading="auto", cmap="viridis", zorder=2)
    ax.contour(X, Y, V_plot, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)

    # Roads
    try:
        roads = gpd.read_file(ROADS_CLIP)
        roads.plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
    except Exception:
        pass

    # Plot all greedy paths converging on MSG
    for i, (src_name, pdata) in enumerate(source_paths.items()):
        color = colors_list[i % len(colors_list)]

        # Greedy path
        if pdata['greedy_path'] and len(pdata['greedy_path']) >= 2:
            C = coords[np.array(pdata['greedy_path'])]
            ax.plot(C[:, 0], C[:, 1], '-', lw=2.5, color=color,
                    label=f"{src_name} ({pdata['greedy_time']:.1f} min)", zorder=5+i)

        # Source marker
        ax.plot(pdata['src_xy'][0], pdata['src_xy'][1], 'o', ms=10, color=color,
                markeredgecolor='black', markeredgewidth=1.5, zorder=8+i)

    # MSG sink marker (larger, prominent)
    ax.plot(msg_xy[0], msg_xy[1], 'r*', ms=22, markeredgecolor='black',
            markeredgewidth=1.5, label='MSG (Sink)', zorder=20)

    ax.set_aspect('equal')
    ax.set_xlabel("x (ft, EPSG:2263)")
    ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title("Fig 5c (Reverse): NYC Converging on MSG\n(All Greedy routes)")
    plt.colorbar(im, ax=ax, label="Potential φ (distance to MSG, scaled)")
    ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    combined_png_path = f"{save_prefix}_combined.png"
    plt.savefig(combined_png_path, dpi=220)
    print(f"Saved: {combined_png_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return results_table, source_paths


# ----------------------- FIG 5D: GW BRIDGE TO COLUMBIA COMMUTE -----------------------
def case_5d_gw_commute(save_prefix="output/fig5d_gw_commute", show=False, detour_point_lonlat=None):
    """
    Fig 5d: Commute from GW Bridge to Columbia with carpooling.
    5000 commuters.

    Uses source-only potential field (no sink):
    - phi = shortest-path distance TO destination
    - Forward greedy routing follows decreasing phi

    Shows:
    - Potential distribution (distance to Columbia)
    - Shortest path comparison: Greedy vs Dijkstra
    - If detour_point is specified: shows detour route GW Bridge -> A -> Columbia
    """
    print("\n" + "="*60)
    print("Fig 5d: GW Bridge to Columbia Commute")
    print("="*60)

    # Load graph and nodes
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)

    # Build time map with minimum weight for duplicate edges
    tmap = {}
    for u, v, w in zip(efrom, eto, etime):
        key = (int(u), int(v))
        if key not in tmap or w < tmap[key]:
            tmap[key] = w

    # Get locations in EPSG:2263
    gw_pt = lonlat_to_2263(GW_BRIDGE_LONLAT)
    gw_xy = np.array([gw_pt.x, gw_pt.y])

    col_pt = lonlat_to_2263(COLUMBIA_LONLAT)
    col_xy = np.array([col_pt.x, col_pt.y])

    # Find source (GW Bridge) and sink (Columbia) nodes
    src = nearest_node(coords, gw_xy)
    dst = nearest_node(coords, col_xy)

    # Find nodes within radius for BCs
    GW_RADIUS = 500.0  # ft
    COL_RADIUS = 500.0  # ft

    dists_to_gw = np.linalg.norm(coords - gw_xy, axis=1)
    dists_to_col = np.linalg.norm(coords - col_xy, axis=1)

    src_nodes = np.where(dists_to_gw <= GW_RADIUS)[0]
    dst_nodes = np.where(dists_to_col <= COL_RADIUS)[0]

    if len(src_nodes) == 0:
        src_nodes = np.array([src])
    if len(dst_nodes) == 0:
        dst_nodes = np.array([dst])

    print(f"GW Bridge source nodes: {len(src_nodes)}")
    print(f"Columbia destination nodes: {len(dst_nodes)}")

    # Forward Greedy - phi = distance TO destination
    phi_to_dest = compute_shortest_path_potential(efrom, eto, etime, dst, V)

    # Grid for interpolation - same approach as solve_continuum_and_plot
    man = gpd.read_file(MAN_2263)
    man_union = make_valid(union_all(list(man.geometry)))
    man_bounds = man_union.bounds

    xg = np.linspace(man_bounds[0] - GRID_BUFFER_FT, man_bounds[2] + GRID_BUFFER_FT, GRID_NX)
    yg = np.linspace(man_bounds[1] - GRID_BUFFER_FT, man_bounds[3] + GRID_BUFFER_FT, GRID_NY)
    X, Y = np.meshgrid(xg, yg)

    # Create Manhattan inside mask
    try:
        from matplotlib.path import Path
        geoms = list(man_union.geoms) if man_union.geom_type == "MultiPolygon" else [man_union]
        P = np.c_[X.ravel(), Y.ravel()]
        inside = np.zeros(P.shape[0], dtype=bool)
        for poly in geoms:
            ex = np.c_[*poly.exterior.xy].astype(float)
            inside |= Path(ex).contains_points(P)
            for ring in poly.interiors:
                ring_path = Path(np.c_[*ring.xy].astype(float))
                inside[ring_path.contains_points(P)] = False
        inside = inside.reshape(Y.shape)
    except Exception:
        pts = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Y.ravel()), crs=man.crs)
        inside = pts.within(man_union).to_numpy().reshape(Y.shape)

    # Interpolate phi to grid for heatmap visualization (use phi_to_dest for viz)
    Phi_lin = griddata(coords, phi_to_dest, (X, Y), method='linear')
    Phi_nn = griddata(coords, phi_to_dest, (X, Y), method='nearest')
    Phi = np.where(np.isfinite(Phi_lin), Phi_lin, Phi_nn)

    # Dijkstra path: GW Bridge -> Columbia (optimal)
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)

    # Forward Greedy path: follow decreasing phi (distance to dest) from source
    greedy_nodes = greedy_path_on_nodes(phi_to_dest, efrom, eto, src, dst, coords=coords)
    greedy_time = path_time(greedy_nodes, tmap)

    print(f"\nDirect route GW Bridge -> Columbia:")
    print(f"  Dijkstra: {dijk_time:.2f} min")
    print(f"  Greedy: {greedy_time:.2f} min")

    # Detour analysis if point A is specified
    detour_results = None
    if detour_point_lonlat is not None:
        detour_pt = lonlat_to_2263(detour_point_lonlat)
        detour_xy = np.array([detour_pt.x, detour_pt.y])
        detour_node = nearest_node(coords, detour_xy)

        # GW Bridge -> A (Dijkstra)
        dijk_gw_a, time_gw_a_dijk = dijkstra_path(coords, efrom, eto, etime, src, detour_node)
        # A -> Columbia (Dijkstra)
        dijk_a_col, time_a_col_dijk = dijkstra_path(coords, efrom, eto, etime, detour_node, dst)

        total_detour_dijk = time_gw_a_dijk + time_a_col_dijk

        # Forward Greedy paths for detour segments
        # For GW -> A: compute shortest-path distance TO detour point
        phi_gw_a = compute_shortest_path_potential(efrom, eto, etime, detour_node, V)
        greedy_gw_a = greedy_path_on_nodes(phi_gw_a, efrom, eto, src, detour_node, coords=coords)
        time_gw_a_greedy = path_time(greedy_gw_a, tmap)

        # For A -> Columbia: compute shortest-path distance TO Columbia
        phi_a_col = compute_shortest_path_potential(efrom, eto, etime, dst, V)
        greedy_a_col = greedy_path_on_nodes(phi_a_col, efrom, eto, detour_node, dst, coords=coords)
        time_a_col_greedy = path_time(greedy_a_col, tmap)

        total_detour_greedy = time_gw_a_greedy + time_a_col_greedy

        detour_results = {
            'detour_xy': detour_xy,
            'detour_node': detour_node,
            'dijk_gw_a': dijk_gw_a,
            'dijk_a_col': dijk_a_col,
            'greedy_gw_a': greedy_gw_a,
            'greedy_a_col': greedy_a_col,
            'time_gw_a_dijk': time_gw_a_dijk,
            'time_a_col_dijk': time_a_col_dijk,
            'time_gw_a_greedy': time_gw_a_greedy,
            'time_a_col_greedy': time_a_col_greedy,
            'total_detour_dijk': total_detour_dijk,
            'total_detour_greedy': total_detour_greedy
        }

        print(f"\nDetour route GW Bridge -> A -> Columbia:")
        print(f"  Dijkstra: {total_detour_dijk:.2f} min (GW->A: {time_gw_a_dijk:.2f}, A->Col: {time_a_col_dijk:.2f})")
        print(f"  Greedy: {total_detour_greedy:.2f} min (GW->A: {time_gw_a_greedy:.2f}, A->Col: {time_a_col_greedy:.2f})")
        print(f"  Detour overhead: {total_detour_dijk - dijk_time:.2f} min (Dijkstra)")

    # ===== PLOT: Same style as solve_continuum_and_plot =====
    Phi_plot = Phi.copy()
    Phi_plot[~inside] = np.nan  # Mask outside Manhattan

    # Normalize to [0,1] for vivid colormap
    vmin, vmax = np.nanmin(Phi_plot), np.nanmax(Phi_plot)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        Phi_plot = (Phi_plot - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(8.8, 11.2))

    # Manhattan boundary
    gpd.GeoDataFrame(geometry=[man_union], crs=man.crs).boundary.plot(
        ax=ax, color="black", linewidth=1.0, zorder=1)

    # Potential heatmap (viridis like original)
    im = ax.pcolormesh(X, Y, Phi_plot, shading="auto", cmap="viridis", zorder=2)

    # Contour lines (black like original)
    ax.contour(X, Y, Phi_plot, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)

    # Roads on top (light gray like original)
    try:
        roads = gpd.read_file(ROADS_CLIP)
        roads.plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
    except Exception:
        pass

    # Direct paths - Dijkstra (magenta), Forward Greedy (cyan)
    if dijk_nodes and len(dijk_nodes) >= 2:
        C = coords[np.array(dijk_nodes)]
        ax.plot(C[:, 0], C[:, 1], 'm-', lw=3, label=f'Dijkstra ({dijk_time:.1f} min)', zorder=5)

    if greedy_nodes and len(greedy_nodes) >= 2:
        C = coords[np.array(greedy_nodes)]
        ax.plot(C[:, 0], C[:, 1], 'c-', lw=3, label=f'Greedy ({greedy_time:.1f} min)', zorder=6)

    # Detour paths if applicable
    if detour_results:
        # GW -> A path (Dijkstra dashed, Greedy solid)
        if detour_results['dijk_gw_a'] and len(detour_results['dijk_gw_a']) >= 2:
            C = coords[np.array(detour_results['dijk_gw_a'])]
            ax.plot(C[:, 0], C[:, 1], 'r--', lw=1.5, alpha=0.7, zorder=5)

        if detour_results['greedy_gw_a'] and len(detour_results['greedy_gw_a']) >= 2:
            C = coords[np.array(detour_results['greedy_gw_a'])]
            ax.plot(C[:, 0], C[:, 1], 'r-', lw=2.5, alpha=0.9,
                    label=f"GW->A Greedy ({detour_results['time_gw_a_greedy']:.1f} min)", zorder=6)

        # A -> Columbia path (Dijkstra dashed, Greedy solid)
        if detour_results['dijk_a_col'] and len(detour_results['dijk_a_col']) >= 2:
            C = coords[np.array(detour_results['dijk_a_col'])]
            ax.plot(C[:, 0], C[:, 1], 'y--', lw=1.5, alpha=0.7, zorder=5)

        if detour_results['greedy_a_col'] and len(detour_results['greedy_a_col']) >= 2:
            C = coords[np.array(detour_results['greedy_a_col'])]
            ax.plot(C[:, 0], C[:, 1], 'y-', lw=2.5, alpha=0.9,
                    label=f"A->Col Greedy ({detour_results['time_a_col_greedy']:.1f} min)", zorder=6)

        # Detour point marker
        ax.plot(detour_results['detour_xy'][0], detour_results['detour_xy'][1],
                's', ms=12, color='orange', markeredgecolor='black', markeredgewidth=1.5,
                label='Pickup point A', zorder=9)

    # Source/Destination markers
    ax.plot(gw_xy[0], gw_xy[1], 'go', ms=12, markeredgecolor='black', markeredgewidth=1.5,
            label='GW Bridge (Source)', zorder=10)
    ax.plot(col_xy[0], col_xy[1], 'ro', ms=12, markeredgecolor='black', markeredgewidth=1.5,
            label='Columbia (Dest)', zorder=10)

    ax.set_aspect('equal')
    ax.set_xlabel("x (ft, EPSG:2263)")
    ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title(f"Fig 5d: GW Bridge to Columbia - {GW_COMMUTERS:,} commuters\n(Dijkstra vs Greedy)")
    plt.colorbar(im, ax=ax, label="Potential φ (scaled)")
    ax.legend(loc='lower right', fontsize=6)

    plt.tight_layout()
    png_path = f"{save_prefix}.png"
    plt.savefig(png_path, dpi=220)
    print(f"\nSaved: {png_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'dijk_time': dijk_time,
        'greedy_time': greedy_time,
        'dijk_nodes': dijk_nodes,
        'greedy_nodes': greedy_nodes,
        'detour': detour_results
    }

# ----------------------- MAIN -----------------------
def main():
    # 1) Align + clip once
    if not os.path.exists(ROADS_CLIP):
        align_and_clip()

    # 2) Build graph + nodes
    if not (os.path.exists(GRAPH_TXT) and os.path.exists(NODES_GDF)):
        build_graph_from_roads()

    # 3) Solve node potentials
    if not os.path.exists(NODES_PHI):
        solve_node_potential()

    # 4) Routes + quick viz
    compute_routes_and_plot()

    # 5) Build K on Voronoi (if needed) and 6) heatmap
    if not (os.path.exists(VORO_OUT) and {"Kxx","Kxy","Kyy"}.issubset(set(gpd.read_file(VORO_OUT).columns))):
        build_K_from_voronoi()
    solve_continuum_and_plot()


def run_fig5c():
    """Run Fig 5c: NBA game dispersal scenario."""
    # Ensure prerequisites
    if not os.path.exists(NODES_PHI):
        main()
    case_5c_nba_game(show=False)


def run_fig5c_arrival():
    """
    Run Fig 5c (Reverse): NBA game arrival scenario.
    People travel FROM distributed sources across NYC TO MSG (sink).
    """
    # Ensure prerequisites
    if not os.path.exists(NODES_PHI):
        main()
    case_5c_nba_arrival(show=False)


def run_fig5d(with_detour=True):
    """
    Run Fig 5d: GW Bridge to Columbia commute.
    If with_detour=True, includes a pickup point A (e.g., Times Square area).
    """
    # Ensure prerequisites
    if not os.path.exists(NODES_PHI):
        main()

    # Example detour point: pickup someone near Times Square
    detour_lonlat = (-73.9855, 40.7580) if with_detour else None
    case_5d_gw_commute(show=False, detour_point_lonlat=detour_lonlat)


if __name__ == "__main__":
    # Ensure basic data is ready
    if not os.path.exists(ROADS_CLIP):
        align_and_clip()
    if not (os.path.exists(GRAPH_TXT) and os.path.exists(NODES_GDF)):
        build_graph_from_roads()
    if not os.path.exists(NODES_PHI):
        solve_node_potential()

    # Run the new figure cases
    print("\n" + "="*70)
    print("Running Fig 5c and Fig 5d simulations")
    print("="*70)

    run_fig5c()
    run_fig5c_arrival()  # Reverse scenario: sources distributed, MSG as sink
    run_fig5d(with_detour=True)
