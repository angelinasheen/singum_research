#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ FULL DROP-IN: Manhattan routes + robust continuum heatmap ============
# What this does:
# 1) reads + aligns data, clips roads to Manhattan (EPSG:2263)
# 2) builds graph with 25 mph edge weights (min) and writes nodes/edges
# 3) solves node potentials (start=1, end=0) and writes nodes_with_phi
# 4) computes Singum (greedy) + Dijkstra routes; writes GeoJSON and PNG
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

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csgraph
from scipy.interpolate import LinearNDInterpolator, griddata

# ----------------------- USER KNOBS -----------------------
TARGET_CRS = "EPSG:2263"

# I/O (adjust if your filenames differ)
ROADS_IN  = "filtered_roads.geojson"               # input raw (full NYC) roads
MAN_IN    = "manhattan_boundary.geojson"           # input raw Manhattan boundary (any CRS)

ROADS_2263 = "filtered_roads_2263.geojson"
MAN_2263   = "manhattan_boundary_2263.geojson"
ROADS_CLIP = "filtered_roads_manhattan_2263.geojson"

# graph + nodes
GRAPH_TXT = "graph_edges.txt"
NODES_GDF = "nodes_points_2263.geojson"
NODES_PHI = "nodes_with_phi_2263.geojson"

# routes + viz
ROUTES_GEOJSON = "manhattan_routes.geojson"
ROUTES_PNG     = "manhattan_routes_viz.png"
ROUTES_SINGUM_PNG = "manhattan_routes_singum_only.png"

# Voronoi
VORO_IN   = "singum_voronoi.geojson"               # must contain NodeID + geometry
VORO_OUT  = "singum_voronoi_with_K.geojson"
VORO_PNG  = "singum_voronoi.png"
K_CENTROID_CSV = "K_centroids_2263.csv"

# continuum heatmap
HEATMAP_PNG = "manhattan_continuum_V.png"

# Start/End (external NodeIDs from the roads attributes)
START_NODEID = 21465
END_NODEID   = 78136

# Continuum BCs
COLUMBIA_LONLAT = (-73.9626, 40.8075)
COLUMBIA_RADIUS_FT = 600.0
TOTAL_FLUX_INTO_COLUMBIA = 5000.0

# ----------------------- LANDMARK LOCATIONS (lon, lat) -----------------------
# Fig 5c & 5d locations
MSG_LONLAT = (-73.9934, 40.7505)           # Madison Square Garden
MSG_RADIUS_FT = 400.0                       # ~400 ft radius for MSG area
GRAND_CENTRAL_LONLAT = (-73.9772, 40.7527)
NYU_LONLAT = (-73.9965, 40.7295)
TIMES_SQUARE_LONLAT = (-73.9855, 40.7580)
PENN_STATION_LONLAT = (-73.9937, 40.7506)
WALL_STREET_LONLAT = (-74.0110, 40.7074)
GW_BRIDGE_LONLAT = (-73.9527, 40.8517)     # George Washington Bridge (Manhattan side)

# For Fig 5c: NBA game destinations
DESTINATIONS = {
    "Grand Central": GRAND_CENTRAL_LONLAT,
    "Columbia": COLUMBIA_LONLAT,
    "NYU": NYU_LONLAT,
    "Times Square": TIMES_SQUARE_LONLAT,
    "Wall Street": WALL_STREET_LONLAT,
}

# For Fig 5d: Commute scenario
GW_COMMUTERS = 5000
NBA_ATTENDEES = 20000

# Grid for continuum
GRID_NX, GRID_NY = 700, 1050
GRID_BUFFER_FT = 250.0

# ----------------------- CONSTANTS -----------------------
MPH_TO_FTPS = 1.46667
SEC_TO_MIN  = 1.0/60.0
CONST_MPH   = 25.0  # all edges 25 mph

# A* style goal heuristic weight for gradient tracing
# 0.0 = pure gradient, higher values = more goal-directed
# Higher values needed when gradient points away from goal (boundary effects)
GOAL_HEURISTIC_WEIGHT = 1.5

# ----------------------- SMALL HELPERS -----------------------
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

def greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=None, max_steps=500000):
    from collections import defaultdict, deque
    nbrs = defaultdict(list)
    for u, v in zip(efrom, eto):
        nbrs[u].append(v); nbrs[v].append(u)

    goal = int(dst)
    tol = 1e-12
    cur = int(src)
    path = [cur]; visited = {cur}

    def dist_goal(u):
        return 0.0 if coords is None else np.linalg.norm(coords[u] - coords[goal])

    steps = 0
    while cur != goal and steps < max_steps:
        steps += 1
        nlist = nbrs[cur]
        lower = [v for v in nlist if phi[v] < phi[cur] - tol]
        if lower:
            best_phi = min(phi[v] for v in lower)
            candidates = [v for v in lower if abs(phi[v] - best_phi) <= tol]
            if len(candidates) > 1:
                nxt = min(candidates, key=dist_goal)
            else:
                nxt = candidates[0]
        else:
            equal = [v for v in nlist if abs(phi[v] - phi[cur]) <= tol]
            if not equal: break
            nxt = min(equal, key=dist_goal)
            if len(path) >= 2 and nxt == path[-2]:
                eq_sorted = sorted(equal, key=dist_goal)
                for cand in eq_sorted:
                    if cand != path[-2]:
                        nxt = cand; break
        if nxt in visited and (len(path) < 2 or nxt != path[-2]):
            break
        path.append(nxt); visited.add(nxt); cur = nxt
    return path

def dijkstra_path(coords, efrom, eto, etime, src, dst):
    rows = np.concatenate([efrom, eto])
    cols = np.concatenate([eto, efrom])
    dat  = np.concatenate([etime, etime])
    N = coords.shape[0]
    adj = csr_matrix((dat,(rows,cols)), shape=(N,N))
    dist, preds = csgraph.dijkstra(adj, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]): return [], np.inf
    path=[]; cur=dst
    while cur!=-9999 and cur!=src:
        path.append(cur); cur=preds[cur]
    path.append(src); path=path[::-1]
    return path, float(dist[dst])

def get_dijkstra_initial_direction(coords, efrom, eto, etime, src, dst):
    """
    Get the direction of Dijkstra's first step from src toward dst.
    Returns a unit vector (dx, dy) or None if path doesn't exist.
    """
    path, _ = dijkstra_path(coords, efrom, eto, etime, src, dst)
    if len(path) < 2:
        return None
    # Direction from src to first waypoint
    direction = coords[path[1]] - coords[path[0]]
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    return direction / norm

def path_time(path, time_map):
    if len(path) < 2: return np.nan
    t=0.0
    for a,b in zip(path[:-1], path[1:]):
        w = time_map.get((a,b)) or time_map.get((b,a))
        if w is None: return np.nan
        t += w
    return t

def lonlat_to_2263(lonlat):
    """Convert (lon, lat) to EPSG:2263 (ft)."""
    man_crs = gpd.read_file(MAN_2263).crs
    return gpd.GeoSeries([Point(lonlat)], crs="EPSG:4326").to_crs(man_crs).iloc[0]

def nearest_node(coords, xy):
    """Find nearest node index to a given (x, y) coordinate."""
    return int(np.argmin(np.linalg.norm(coords - np.array(xy)[None,:], axis=1)))

def gradient_trace(gradx_interp, grady_interp, coords, src, dst, steplength=50.0, max_steps=50000,
                   initial_direction=None, initial_kick_steps=5, goal_heuristic_weight=0.0):
    """
    Follow continuous -grad(phi) to trace a path from src to dst.
    Adapted for Manhattan scale (feet).

    Parameters:
    -----------
    initial_direction : array-like or None
        If provided, use this direction for the first `initial_kick_steps` steps
        before switching to gradient following. This helps when the gradient
        at the source is ambiguous (e.g., at a source point where phi is maximum).
    initial_kick_steps : int
        Number of steps to follow the initial_direction before using gradient.
    goal_heuristic_weight : float
        A* style heuristic weight. Blends goal direction into the gradient:
        direction = -grad(phi) + goal_heuristic_weight * (goal - current_pos)
        Set to 0 for pure gradient following. Try 0.1-0.5 for goal bias.
    """
    from scipy.interpolate import RegularGridInterpolator

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    pt = coords[int(src)].astype(float).copy()
    goal = coords[int(dst)].astype(float)
    path_xy = [pt.copy()]

    # Normalize initial direction if provided
    init_dir = None
    if initial_direction is not None:
        init_dir = np.array(initial_direction, float)
        norm = np.linalg.norm(init_dir)
        if norm > 1e-12:
            init_dir = init_dir / norm
        else:
            init_dir = None

    for step_idx in range(int(max_steps)):
        # Use initial direction for first few steps (Dijkstra kick)
        if init_dir is not None and step_idx < initial_kick_steps:
            pt = pt + steplength * init_dir
        else:
            gx = gradx_interp((pt[1], pt[0]))
            gy = grady_interp((pt[1], pt[0]))
            if not np.isfinite(gx) or not np.isfinite(gy):
                break
            gvec = np.array([gx, gy], float)
            gnrm = np.linalg.norm(gvec)

            # Compute goal direction (A* heuristic component)
            goal_dir = goal - pt
            goal_dist = np.linalg.norm(goal_dir)

            if goal_dist < steplength:
                path_xy.append(goal.copy())
                break

            goal_dir = goal_dir / goal_dist  # Unit vector toward goal

            if gnrm < 1e-12:
                # Near zero gradient, head directly to goal
                pt = pt + steplength * goal_dir
            else:
                # Blend gradient direction with goal heuristic (A* style)
                grad_dir = -gvec / gnrm  # Unit vector in -grad(phi) direction

                # Combined direction: gradient + λ * goal_direction
                combined = grad_dir + goal_heuristic_weight * goal_dir
                combined_norm = np.linalg.norm(combined)

                if combined_norm > 1e-12:
                    combined = combined / combined_norm
                else:
                    combined = goal_dir

                pt = pt + steplength * combined

        pt[0] = np.clip(pt[0], x_min, x_max)
        pt[1] = np.clip(pt[1], y_min, y_max)
        path_xy.append(pt.copy())

        if np.linalg.norm(pt - goal) <= 2 * steplength:
            path_xy.append(goal.copy())
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

# ----------------------- DATA PREP -----------------------
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

    # symmetric Laplacian from undirected conductances
    Gpair = defaultdict(float)
    for u,v,t in zip(efrom,eto,etime):
        if t > 0:
            a,b = (u,v) if u<v else (v,u)
            Gpair[(a,b)] += 1.0/max(t,1e-12)
    rows, cols, data = [], [], []
    for (a,b), G in Gpair.items():
        rows += [a,b,a,b]; cols += [a,b,b,a]; data += [G, G, -G, -G]
    L = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(V,V))

    node_index = {nid:i for i,nid in enumerate(list_nodes)}
    s = node_index[START_NODEID]; t = node_index[END_NODEID]
    ML, b = apply_dirichlet_row_replace(L, [s], 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, [t], 0.0); b += b2
    phi = solve_with_component_pinning(ML, b, pinned_nodes=np.r_[s,t])

    nodes["phi"] = phi
    nodes.to_file(NODES_PHI, driver="GeoJSON")

def compute_routes_and_plot():
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    phi    = nodes["phi"].to_numpy(float)

    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)
    assert V == coords.shape[0] == phi.shape[0], "Node count mismatch."

    idx = {nid:i for i,nid in enumerate(list_nodes)}
    src = idx[START_NODEID]; dst = idx[END_NODEID]
    tmap = {(u,v):w for u,v,w in zip(efrom,eto,etime)}

    sing_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)
    sing_time = path_time(sing_nodes, tmap)

    # write routes
    def to_line(path): return LineString([coords[i] for i in path]) if len(path)>=2 else None
    out = gpd.GeoDataFrame(
        {"name": ["singum","dijkstra"],
         "time_min":[float(sing_time), float(dijk_time)],
         "start_id":[START_NODEID]*2, "end_id":[END_NODEID]*2},
        geometry=[to_line(sing_nodes), to_line(dijk_nodes)],
        crs=gpd.read_file(MAN_2263).crs
    ).dropna(subset=["geometry"])
    out.to_file(ROUTES_GEOJSON, driver="GeoJSON")

    # viz with roads + legend
    fig, ax = plt.subplots(figsize=(8.2, 10.5))
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

    if len(sing_nodes)>=2:
        C = coords[np.array(sing_nodes)]
        ax.plot(C[:,0], C[:,1], '-', lw=3, color='cyan', label=f"Singum greedy ({sing_time:.1f} min)")
    if len(dijk_nodes)>=2:
        C = coords[np.array(dijk_nodes)]
        ax.plot(C[:,0], C[:,1], '-', lw=3, color='magenta', label=f"Dijkstra ({dijk_time:.1f} min)")

    ax.plot(coords[src,0], coords[src,1], 'go', ms=8, label="Start")
    ax.plot(coords[dst,0], coords[dst,1], 'ro', ms=8, label="End")
    ax.set_aspect('equal'); ax.set_xlabel("x (ft, EPSG:2263)"); ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title("Manhattan: Singum (greedy) vs Dijkstra (25 mph edges)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(ROUTES_PNG, dpi=220)
    print(f"Singum time: {sing_time:.2f} min  |  Dijkstra time: {dijk_time:.2f} min")
    print("Saved:", ROUTES_PNG)

    # singum-only PNG
    fig2, ax2 = plt.subplots(figsize=(8.2, 10.5))
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
    if len(sing_nodes)>=2:
        C = coords[np.array(sing_nodes)]
        ax2.plot(C[:,0], C[:,1], '-', lw=3, color='cyan', label=f"Singum greedy ({sing_time:.1f} min)")
    ax2.plot(coords[src,0], coords[src,1], 'go', ms=8, label="Start")
    ax2.plot(coords[dst,0], coords[dst,1], 'ro', ms=8, label="End")
    ax2.set_aspect('equal'); ax2.set_xlabel("x (ft, EPSG:2263)"); ax2.set_ylabel("y (ft, EPSG:2263)")
    ax2.set_title("Manhattan: Singum (greedy) route")
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

    # undirected conductance
    Gpair = defaultdict(float)
    for u,v,t in zip(efrom,eto,etime):
        if t > 0:
            a,b = (u,v) if u<v else (v,u)
            Gpair[(a,b)] += 1.0/t

    Kxx = np.zeros(V); Kxy = np.zeros(V); Kyy = np.zeros(V)
    for (a,b), G in Gpair.items():
        pa, pb = coords[a], coords[b]
        d = pb - pa
        L = np.linalg.norm(d)
        if L <= 0 or G <= 0: continue
        n = d/L
        factor = 0.5*(L*L)*G
        nx, ny = float(n[0]), float(n[1])
        for u in (a,b):
            Kxx[u] += factor*nx*nx
            Kxy[u] += factor*nx*ny
            Kyy[u] += factor*ny*ny

    nodeid_to_idx = {nid:i for i,nid in enumerate(list_nodes)}
    idx_to_nodeid = np.array(list_nodes, int)
    area = np.zeros(V)
    vor_keep = vor[vor["NodeID"].isin(nodeid_to_idx)].copy()
    area_idx = vor_keep["NodeID"].map(nodeid_to_idx).to_numpy(int)
    area[area_idx] = vor_keep["area_ft2"].to_numpy(float)
    safe = np.where(area>0, area, np.nan)

    Kxx /= safe; Kxy /= safe; Kyy /= safe
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

    # Grid metrics
    nxg = X.shape[1]; nyg = X.shape[0]
    dx = (X[0,-1]-X[0,0])/(nxg-1); dy = (Y[-1,0]-Y[0,0])/(nyg-1)
    N  = nxg*nyg

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

    # 3) Columbia ring → Neumann flux
    man_crs = gpd.read_file(MAN_2263).crs
    def lonlat_to_2263(lonlat):
        return gpd.GeoSeries([Point(lonlat)], crs="EPSG:4326").to_crs(man_crs).iloc[0]

    col_pt = lonlat_to_2263(COLUMBIA_LONLAT)
    col_circle = col_pt.buffer(COLUMBIA_RADIUS_FT)
    L_col = float(col_circle.exterior.length)
    from shapely.prepared import prep
    ring = prep(col_circle.boundary.buffer(0.6*max(dx,dy)))
    C_mask = np.zeros_like(inside, dtype=bool)
    xs = X[0,:]; ys = Y[:,0]
    for j in range(0, nyg, max(1, nyg//200)):
        pts = [Point(x, ys[j]) for x in xs]
        mask = np.array([ring.contains(pt) for pt in pts])
        C_mask[j, mask] = True
    for i in range(0, nxg, max(1, nxg//200)):
        pts = [Point(xs[i], y) for y in ys]
        mask = np.array([ring.contains(pt) for pt in pts])
        C_mask[mask, i] = True
    C_mask &= inside

    # 4) Assemble finite-volume system (anisotropic)
    A_rows, A_cols, A_data = [], [], []
    bvec = np.zeros(N, float)
    qn   = -float(TOTAL_FLUX_INTO_COLUMBIA)/max(L_col,1e-12)

    def I(i,j): return j*nxg + i

    for j in range(nyg):
        for i in range(nxg):
            p = I(i,j)

            # Outside domain
            if not inside[j,i]:
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=0.0
                continue

            # Dirichlet shoreline: south=1, north=0
            if D1_mask[j,i]:
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=1.0
                continue
            if D0_mask[j,i]:
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=0.0
                continue

            # Interior cell
            kxx = Kxxg[j,i]; kxy = Kxyg[j,i]; kyy = Kyyg[j,i]
            if not (np.isfinite(kxx) and np.isfinite(kxy) and np.isfinite(kyy)):
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=0.0
                continue

            diag = 0.0
            for di,dj,Lf,ds,nvec in [
                (-1,0,dy,dx,np.array([-1.0,0.0])),
                ( 1,0,dy,dx,np.array([ 1.0,0.0])),
                ( 0,-1,dx,dy,np.array([ 0.0,-1.0])),
                ( 0, 1,dx,dy,np.array([ 0.0, 1.0])),
            ]:
                ii, jj = i+di, j+dj
                if ii<0 or ii>=nxg or jj<0 or jj>=nyg or (not inside[jj,ii]):
                    continue
                kxx2 = 0.5*(kxx + Kxxg[jj,ii])
                kxy2 = 0.5*(kxy + Kxyg[jj,ii])
                kyy2 = 0.5*(kyy + Kyyg[jj,ii])
                if not (np.isfinite(kxx2) and np.isfinite(kxy2) and np.isfinite(kyy2)):
                    continue
                Kmat = np.array([[kxx2,kxy2],[kxy2,kyy2]])
                nKn  = float(nvec @ Kmat @ nvec)
                if nKn <= 0: 
                    continue
                G = nKn * (Lf/ds)
                diag += G
                A_rows.append(p); A_cols.append(I(ii,jj)); A_data.append(-G)

            if diag <= 0.0:
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=0.0
            else:
                A_rows.append(p); A_cols.append(p); A_data.append(diag)
                if C_mask[j,i]:
                    perimeter_share = (2*dx + 2*dy)/4.0
                    bvec[p] += qn * perimeter_share

    A = csr_matrix((np.array(A_data), (np.array(A_rows), np.array(A_cols))), shape=(N,N))

    # Tiny diagonal regularization to kill any residual nullspace
    eps = 1e-9
    A = A + csr_matrix((np.full(N, eps), (np.arange(N), np.arange(N))), shape=(N,N))

    # Sanitize numeric issues
    if not np.isfinite(A.data).all(): A.data[~np.isfinite(A.data)] = 0.0
    if not np.isfinite(bvec).all():   bvec[~np.isfinite(bvec)] = 0.0

    V = spsolve(A, bvec).reshape(Y.shape)

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
def case_5c_nba_game(save_prefix="fig5c_nba", show=False):
    """
    Fig 5c: NBA game over at Madison Square Garden.
    20,000 people leave MSG (source) and disperse across Manhattan (distributed sink).

    Source: q = +20,000 / A_MSG  at MSG
    Sink:   q = -20,000 / (A_Manhattan - A_MSG) uniformly elsewhere

    Shows potential distribution, flow pattern, and compares gradient vs Dijkstra
    paths to several destinations.
    """
    from scipy.interpolate import RegularGridInterpolator

    print("\n" + "="*60)
    print("Fig 5c: NBA Game Over - MSG dispersal scenario")
    print("="*60)

    # Load graph and nodes
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    N_nodes, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)

    # Build time map
    tmap = {(u, v): w for u, v, w in zip(efrom, eto, etime)}
    for u, v, w in zip(efrom, eto, etime):
        tmap[(v, u)] = w

    # Build symmetric Laplacian (conductance = 1/time)
    Gpair = defaultdict(float)
    for u, v, t in zip(efrom, eto, etime):
        if t > 0:
            a, b = (u, v) if u < v else (v, u)
            Gpair[(a, b)] += 1.0 / max(t, 1e-12)

    rows, cols, data = [], [], []
    for (a, b), G in Gpair.items():
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        data += [G, G, -G, -G]
    L = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N_nodes, N_nodes))

    # Get MSG location in EPSG:2263
    msg_pt = lonlat_to_2263(MSG_LONLAT)
    msg_xy = np.array([msg_pt.x, msg_pt.y])

    # Get Manhattan boundary and create inside mask
    man = gpd.read_file(MAN_2263)
    man_union = make_valid(union_all(list(man.geometry)))
    A_manhattan = man_union.area  # sq ft

    # MSG area (circular approximation)
    A_msg = np.pi * MSG_RADIUS_FT**2

    # Find nodes within MSG radius (source nodes)
    dists_to_msg = np.linalg.norm(coords - msg_xy, axis=1)
    src_nodes = np.where(dists_to_msg <= MSG_RADIUS_FT)[0]
    if len(src_nodes) == 0:
        src_nodes = np.array([nearest_node(coords, msg_xy)])

    print(f"MSG source nodes: {len(src_nodes)}")
    print(f"A_Manhattan = {A_manhattan:.0f} sq ft")
    print(f"A_MSG = {A_msg:.0f} sq ft")

    # Grid setup for plotting
    man_bounds = man_union.bounds
    xg = np.linspace(man_bounds[0] - GRID_BUFFER_FT, man_bounds[2] + GRID_BUFFER_FT, GRID_NX)
    yg = np.linspace(man_bounds[1] - GRID_BUFFER_FT, man_bounds[3] + GRID_BUFFER_FT, GRID_NY)
    X, Y = np.meshgrid(xg, yg)
    dy = yg[1] - yg[0]
    dx = xg[1] - xg[0]

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

    # Compute paths to each destination
    # KEY: For each destination, we need a SEPARATE potential field (MSG=1, dest=0)
    # This ensures the gradient points from MSG toward that specific destination
    results_table = []
    destination_paths = {}
    phi_for_plot = None  # Will store one phi field for visualization

    for dest_name, dest_lonlat in DESTINATIONS.items():
        dest_pt = lonlat_to_2263(dest_lonlat)
        dest_xy = np.array([dest_pt.x, dest_pt.y])
        dst = nearest_node(coords, dest_xy)

        # Find destination nodes within a radius
        dists_to_dst = np.linalg.norm(coords - dest_xy, axis=1)
        dst_nodes = np.where(dists_to_dst <= 500.0)[0]
        if len(dst_nodes) == 0:
            dst_nodes = np.array([dst])

        # Solve potential field: MSG=1 (source), this destination=0 (sink)
        ML_dest, b_dest = apply_dirichlet_row_replace(L, src_nodes, 1.0)
        ML_dest, b2_dest = apply_dirichlet_row_replace(ML_dest, dst_nodes, 0.0)
        b_dest += b2_dest

        phi_dest = solve_with_component_pinning(ML_dest, b_dest, pinned_nodes=np.r_[src_nodes, dst_nodes])

        # Store first phi for plotting
        if phi_for_plot is None:
            phi_for_plot = phi_dest.copy()

        # Interpolate phi to grid for gradient computation
        Phi_dest_lin = griddata(coords, phi_dest, (X, Y), method='linear')
        Phi_dest_nn = griddata(coords, phi_dest, (X, Y), method='nearest')
        Phi_dest = np.where(np.isfinite(Phi_dest_lin), Phi_dest_lin, Phi_dest_nn)

        # Compute gradients for this destination
        dPhidy_dest, dPhidx_dest = np.gradient(Phi_dest, dy, dx, edge_order=2)
        dPhidx_dest = np.nan_to_num(dPhidx_dest, nan=0.0)
        dPhidy_dest = np.nan_to_num(dPhidy_dest, nan=0.0)

        gradx_interp = RegularGridInterpolator((yg, xg), dPhidx_dest, method='linear',
                                                bounds_error=False, fill_value=0.0)
        grady_interp = RegularGridInterpolator((yg, xg), dPhidy_dest, method='linear',
                                                bounds_error=False, fill_value=0.0)

        # Dijkstra path
        dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)

        # Get Dijkstra's first step direction for gradient initialization
        init_dir = get_dijkstra_initial_direction(coords, efrom, eto, etime, src, dst)

        # Gradient path - follow -grad(phi) from MSG (high=1) to destination (low=0)
        # Use Dijkstra's first step as initial direction to avoid swerving at source
        # Use A* style goal heuristic to help through flat potential regions
        grad_xy = gradient_trace(gradx_interp, grady_interp, coords, src, dst,
                                  steplength=50.0, max_steps=50000,
                                  initial_direction=init_dir, initial_kick_steps=10,
                                  goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)
        snapped_nodes = snap_path_to_network(grad_xy, coords, efrom, eto, src, dst)
        grad_time = path_time(snapped_nodes, tmap)

        results_table.append({
            'Destination': dest_name,
            'Dijkstra (min)': f"{dijk_time:.2f}",
            'Gradient (min)': f"{grad_time:.2f}" if np.isfinite(grad_time) else "N/A",
            'Diff (%)': f"{100*(grad_time - dijk_time)/dijk_time:.1f}" if np.isfinite(grad_time) and dijk_time > 0 else "N/A"
        })

        destination_paths[dest_name] = {
            'dijkstra': dijk_nodes,
            'gradient': snapped_nodes,
            'gradient_raw_xy': grad_xy,  # Raw gradient path (x,y coords) for debugging
            'dijk_time': dijk_time,
            'grad_time': grad_time,
            'dest_xy': dest_xy,
            'dst_node': dst
        }

        print(f"  {dest_name}: Dijkstra={dijk_time:.2f} min, Gradient={grad_time:.2f} min")

    # Print results table
    print("\n" + "-"*60)
    print("Travel Time Comparison Table:")
    print("-"*60)
    print(f"{'Destination':<15} {'Dijkstra (min)':<15} {'Gradient (min)':<15} {'Diff (%)':<10}")
    print("-"*60)
    for row in results_table:
        print(f"{row['Destination']:<15} {row['Dijkstra (min)']:<15} {row['Gradient (min)']:<15} {row['Diff (%)']:<10}")
    print("-"*60)

    # ===== PLOT: Separate graph for each destination =====
    colors_list = ['cyan', 'magenta', 'red', 'yellow', 'lime']

    for i, (dest_name, pdata) in enumerate(destination_paths.items()):
        color = colors_list[i % len(colors_list)]

        # Get the phi field for this specific destination
        dest_pt = lonlat_to_2263(DESTINATIONS[dest_name])
        dest_xy = np.array([dest_pt.x, dest_pt.y])
        dst_node = nearest_node(coords, dest_xy)

        # Find destination nodes within a radius
        dists_to_dst = np.linalg.norm(coords - dest_xy, axis=1)
        dst_nodes = np.where(dists_to_dst <= 500.0)[0]
        if len(dst_nodes) == 0:
            dst_nodes = np.array([dst_node])

        # Solve potential field for this destination
        ML_dest, b_dest = apply_dirichlet_row_replace(L, src_nodes, 1.0)
        ML_dest, b2_dest = apply_dirichlet_row_replace(ML_dest, dst_nodes, 0.0)
        b_dest += b2_dest
        phi_dest = solve_with_component_pinning(ML_dest, b_dest, pinned_nodes=np.r_[src_nodes, dst_nodes])

        # Interpolate phi to grid
        Phi_lin = griddata(coords, phi_dest, (X, Y), method='linear')
        Phi_nn = griddata(coords, phi_dest, (X, Y), method='nearest')
        Phi = np.where(np.isfinite(Phi_lin), Phi_lin, Phi_nn)

        Phi_plot = Phi.copy()
        Phi_plot[~inside] = np.nan

        # Normalize to [0,1]
        vmin, vmax = np.nanmin(Phi_plot), np.nanmax(Phi_plot)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            Phi_plot = (Phi_plot - vmin) / (vmax - vmin)

        # Compute gradient for this destination
        dPhidy_dest, dPhidx_dest = np.gradient(Phi, dy, dx, edge_order=2)
        dPhidx_dest = np.nan_to_num(dPhidx_dest, nan=0.0)
        dPhidy_dest = np.nan_to_num(dPhidy_dest, nan=0.0)

        gradx_interp = RegularGridInterpolator((yg, xg), dPhidx_dest, method='linear',
                                                bounds_error=False, fill_value=0.0)
        grady_interp = RegularGridInterpolator((yg, xg), dPhidy_dest, method='linear',
                                                bounds_error=False, fill_value=0.0)

        # Get Dijkstra's initial direction for this destination
        init_dir_plot = get_dijkstra_initial_direction(coords, efrom, eto, etime, src, dst_node)

        # Get raw gradient path (not snapped to nodes) for debugging
        # Use Dijkstra's first step as initial direction to avoid swerving at source
        # Use A* style goal heuristic to help through flat potential regions
        grad_xy_raw = gradient_trace(gradx_interp, grady_interp, coords, src, dst_node,
                                      steplength=50.0, max_steps=50000,
                                      initial_direction=init_dir_plot, initial_kick_steps=10,
                                      goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)

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

        # Plot raw gradient path (continuous, not snapped to nodes) - FOR DEBUGGING
        if len(grad_xy_raw) >= 2:
            ax.plot(grad_xy_raw[:, 0], grad_xy_raw[:, 1], '-', lw=3, color=color,
                    label=f"Gradient path (raw)", zorder=6)

        # Dijkstra path (dashed line)
        if pdata['dijkstra']:
            C = coords[np.array(pdata['dijkstra'])]
            ax.plot(C[:, 0], C[:, 1], '--', lw=2, color='magenta', alpha=0.7,
                    label=f"Dijkstra ({pdata['dijk_time']:.1f} min)", zorder=5)

        # Destination marker
        ax.plot(pdata['dest_xy'][0], pdata['dest_xy'][1], 'o', ms=12, color=color,
                markeredgecolor='black', markeredgewidth=1.5, label=dest_name, zorder=8)

        # MSG source marker
        ax.plot(msg_xy[0], msg_xy[1], 'g*', ms=18, markeredgecolor='black',
                markeredgewidth=1, label='MSG (Source)', zorder=10)
        circle = plt.Circle((msg_xy[0], msg_xy[1]), MSG_RADIUS_FT * 3, fill=False,
                             color='green', linewidth=2, linestyle='--', zorder=9)
        ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlabel("x (ft, EPSG:2263)")
        ax.set_ylabel("y (ft, EPSG:2263)")
        ax.set_title(f"Fig 5c: MSG to {dest_name}\n(Gradient path - raw, not snapped to nodes)")
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

# ----------------------- FIG 5D: GW BRIDGE TO COLUMBIA COMMUTE -----------------------
def case_5d_gw_commute(save_prefix="fig5d_gw_commute", show=False, detour_point_lonlat=None):
    """
    Fig 5d: Commute from GW Bridge to Columbia with carpooling.
    5000 commuters.

    Shows:
    - Potential distribution from GW Bridge (source) to Columbia (sink)
    - Shortest path comparison: Gradient vs Dijkstra
    - If detour_point is specified: shows detour route GW Bridge -> A -> Columbia
    """
    from scipy.interpolate import RegularGridInterpolator

    print("\n" + "="*60)
    print("Fig 5d: GW Bridge to Columbia Commute")
    print("="*60)

    # Load graph and nodes
    nodes = gpd.read_file(NODES_PHI).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)

    # Build time map
    tmap = {(u, v): w for u, v, w in zip(efrom, eto, etime)}
    for u, v, w in zip(efrom, eto, etime):
        tmap[(v, u)] = w

    # Build symmetric Laplacian
    Gpair = defaultdict(float)
    for u, v, t in zip(efrom, eto, etime):
        if t > 0:
            a, b = (u, v) if u < v else (v, u)
            Gpair[(a, b)] += 1.0 / max(t, 1e-12)

    rows, cols, data = [], [], []
    for (a, b), G in Gpair.items():
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        data += [G, G, -G, -G]
    L = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(V, V))

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
    print(f"Columbia sink nodes: {len(dst_nodes)}")

    # Apply Dirichlet BCs: GW Bridge = 1, Columbia = 0
    ML, b = apply_dirichlet_row_replace(L, src_nodes, 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, dst_nodes, 0.0)
    b += b2

    phi = solve_with_component_pinning(ML, b, pinned_nodes=np.r_[src_nodes, dst_nodes])

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

    # Interpolate phi to grid
    Phi_lin = griddata(coords, phi, (X, Y), method='linear')
    Phi_nn = griddata(coords, phi, (X, Y), method='nearest')
    Phi = np.where(np.isfinite(Phi_lin), Phi_lin, Phi_nn)

    # Compute gradients
    dy = yg[1] - yg[0]
    dx = xg[1] - xg[0]
    dPhidy, dPhidx = np.gradient(Phi, dy, dx, edge_order=2)

    # Replace any NaN in gradients with 0
    dPhidx = np.nan_to_num(dPhidx, nan=0.0)
    dPhidy = np.nan_to_num(dPhidy, nan=0.0)

    # Linear interpolation (more robust)
    gradx_interp = RegularGridInterpolator((yg, xg), dPhidx, method='linear',
                                            bounds_error=False, fill_value=0.0)
    grady_interp = RegularGridInterpolator((yg, xg), dPhidy, method='linear',
                                            bounds_error=False, fill_value=0.0)

    # Dijkstra path: GW Bridge -> Columbia
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)

    # Get Dijkstra's initial direction
    init_dir = get_dijkstra_initial_direction(coords, efrom, eto, etime, src, dst)

    # Gradient path: GW Bridge -> Columbia
    # Use Dijkstra's first step as initial direction to avoid swerving at source
    # Use A* style goal heuristic to help through flat potential regions
    grad_xy = gradient_trace(gradx_interp, grady_interp, coords, src, dst,
                              steplength=50.0, max_steps=50000,
                              initial_direction=init_dir, initial_kick_steps=10,
                              goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)
    snapped_nodes = snap_path_to_network(grad_xy, coords, efrom, eto, src, dst)
    grad_time = path_time(snapped_nodes, tmap)

    print(f"\nDirect route GW Bridge -> Columbia:")
    print(f"  Dijkstra: {dijk_time:.2f} min")
    print(f"  Gradient: {grad_time:.2f} min")

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

        # Gradient path for detour
        # Need separate potential field with A as intermediate
        # Simplified: use streamline from GW to A, then A to Columbia
        init_dir_gw_a = get_dijkstra_initial_direction(coords, efrom, eto, etime, src, detour_node)
        grad_gw_a = gradient_trace(gradx_interp, grady_interp, coords, src, detour_node,
                                    steplength=100.0, max_steps=20000,
                                    initial_direction=init_dir_gw_a, initial_kick_steps=10,
                                    goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)
        snapped_gw_a = snap_path_to_network(grad_gw_a, coords, efrom, eto, src, detour_node)
        time_gw_a_grad = path_time(snapped_gw_a, tmap)

        # For A -> Columbia, we can use the same field (following -grad phi)
        init_dir_a_col = get_dijkstra_initial_direction(coords, efrom, eto, etime, detour_node, dst)
        grad_a_col = gradient_trace(gradx_interp, grady_interp, coords, detour_node, dst,
                                     steplength=100.0, max_steps=20000,
                                     initial_direction=init_dir_a_col, initial_kick_steps=10,
                                     goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)
        snapped_a_col = snap_path_to_network(grad_a_col, coords, efrom, eto, detour_node, dst)
        time_a_col_grad = path_time(snapped_a_col, tmap)

        total_detour_grad = time_gw_a_grad + time_a_col_grad

        detour_results = {
            'detour_xy': detour_xy,
            'detour_node': detour_node,
            'dijk_gw_a': dijk_gw_a,
            'dijk_a_col': dijk_a_col,
            'grad_gw_a': snapped_gw_a,
            'grad_a_col': snapped_a_col,
            'time_gw_a_dijk': time_gw_a_dijk,
            'time_a_col_dijk': time_a_col_dijk,
            'time_gw_a_grad': time_gw_a_grad,
            'time_a_col_grad': time_a_col_grad,
            'total_detour_dijk': total_detour_dijk,
            'total_detour_grad': total_detour_grad
        }

        print(f"\nDetour route GW Bridge -> A -> Columbia:")
        print(f"  Dijkstra: {total_detour_dijk:.2f} min (GW->A: {time_gw_a_dijk:.2f}, A->Col: {time_a_col_dijk:.2f})")
        print(f"  Gradient: {total_detour_grad:.2f} min (GW->A: {time_gw_a_grad:.2f}, A->Col: {time_a_col_grad:.2f})")
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

    # Direct paths - Raw gradient path (solid cyan) and Dijkstra (dashed magenta)
    # Plot raw gradient path (continuous, NOT snapped to nodes) for debugging
    if len(grad_xy) >= 2:
        ax.plot(grad_xy[:, 0], grad_xy[:, 1], 'c-', lw=3, label=f'Gradient (raw path)', zorder=6)

    if dijk_nodes:
        C = coords[np.array(dijk_nodes)]
        ax.plot(C[:, 0], C[:, 1], 'm--', lw=2, label=f'Dijkstra ({dijk_time:.1f} min)', zorder=5)

    # Detour paths if applicable (showing raw gradient paths for debugging)
    if detour_results:
        # GW -> A path: raw gradient trace with Dijkstra kick and goal heuristic
        init_dir_gw_a_raw = get_dijkstra_initial_direction(coords, efrom, eto, etime, src, detour_results['detour_node'])
        grad_gw_a_raw = gradient_trace(gradx_interp, grady_interp, coords, src, detour_results['detour_node'],
                                        steplength=50.0, max_steps=50000,
                                        initial_direction=init_dir_gw_a_raw, initial_kick_steps=10,
                                        goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)
        if len(grad_gw_a_raw) >= 2:
            ax.plot(grad_gw_a_raw[:, 0], grad_gw_a_raw[:, 1], 'r-', lw=2.5, alpha=0.9,
                    label='GW->A (raw gradient)', zorder=6)

        if detour_results['dijk_gw_a']:
            C = coords[np.array(detour_results['dijk_gw_a'])]
            ax.plot(C[:, 0], C[:, 1], 'r--', lw=1.5, alpha=0.7, zorder=5)

        # A -> Columbia path: raw gradient trace with Dijkstra kick and goal heuristic
        init_dir_a_col_raw = get_dijkstra_initial_direction(coords, efrom, eto, etime, detour_results['detour_node'], dst)
        grad_a_col_raw = gradient_trace(gradx_interp, grady_interp, coords, detour_results['detour_node'], dst,
                                         steplength=50.0, max_steps=50000,
                                         initial_direction=init_dir_a_col_raw, initial_kick_steps=10,
                                         goal_heuristic_weight=GOAL_HEURISTIC_WEIGHT)
        if len(grad_a_col_raw) >= 2:
            ax.plot(grad_a_col_raw[:, 0], grad_a_col_raw[:, 1], 'y-', lw=2.5, alpha=0.9,
                    label='A->Columbia (raw gradient)', zorder=6)

        if detour_results['dijk_a_col']:
            C = coords[np.array(detour_results['dijk_a_col'])]
            ax.plot(C[:, 0], C[:, 1], 'y--', lw=1.5, alpha=0.7, zorder=5)

        # Detour point marker
        ax.plot(detour_results['detour_xy'][0], detour_results['detour_xy'][1],
                's', ms=12, color='orange', markeredgecolor='black', markeredgewidth=1.5,
                label='Pickup point A', zorder=9)

    # Source/Destination markers
    ax.plot(gw_xy[0], gw_xy[1], 'go', ms=12, markeredgecolor='black', markeredgewidth=1.5,
            label='GW Bridge (Source)', zorder=10)
    ax.plot(col_xy[0], col_xy[1], 'ro', ms=12, markeredgecolor='black', markeredgewidth=1.5,
            label='Columbia (Sink)', zorder=10)

    ax.set_aspect('equal')
    ax.set_xlabel("x (ft, EPSG:2263)")
    ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title(f"Fig 5d: GW Bridge to Columbia - {GW_COMMUTERS:,} commuters\n(solid=Raw gradient path, dashed=Dijkstra)")
    plt.colorbar(im, ax=ax, label="Potential φ (scaled)")
    ax.legend(loc='lower right', fontsize=7)

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
        'grad_time': grad_time,
        'dijk_nodes': dijk_nodes,
        'grad_nodes': snapped_nodes,
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
    run_fig5d(with_detour=True)
