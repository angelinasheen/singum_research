import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.sparse import csgraph
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle


MPH_TO_MPS = 0.44704
lp0 = 0.005
RADIUS = 20*lp0
steplen = 3.86e-4

# -------------------------- general helpers --------------------------

def nan_gaussian_blur(Z, sigma_pix=0.8):
    """NaN-aware Gaussian blur (for prettier fields)."""
    from scipy.signal import convolve2d
    if sigma_pix <= 0 or Z is None:
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

def nearest_node(coords, xy):
    return int(np.argmin(np.linalg.norm(coords - np.array(xy)[None,:], axis=1)))

# -------------------------- geometry tests for obstacles --------------------------

def seg_intersects_circle(p0, p1, center, radius):
    """True if the segment p0-p1 intersects or lies fully within a circle."""
    c = np.asarray(center, float); a = np.asarray(p0, float); b = np.asarray(p1, float)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 0:
        return np.linalg.norm(a - c) <= radius
    t = np.clip(np.dot(c - a, ab) / denom, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(closest - c) <= radius

def line_intersects_segment(p0, p1, a, b):
    def cross(u,v): return u[0]*v[1]-u[1]*v[0]
    p0 = np.asarray(p0); p1=np.asarray(p1); a=np.asarray(a); b=np.asarray(b)
    d1 = p1-p0; d2 = b-a
    D = cross(d1,d2)
    if np.isclose(D,0.0): return False
    t = cross(a-p0, d2)/D
    u = cross(a-p0, d1)/D
    return (0<=t<=1) and (0<=u<=1)

# -------------------------- lattices --------------------------

def build_tri_lattice(L=2.0, W=np.sqrt(3.0), lp0=lp0):
    """hexagonal lattice with spacing 2*lp0"""
    dx = 2*lp0
    dy = np.sqrt(3.0)*lp0
    nx = int(round(L/dx)+1)
    ny = int(round(W/dy)+1)
    N  = nx*ny
    coords = np.zeros((N,2), float)
    def idx(i,j): return j*nx+i
    for j in range(ny):
        for i in range(nx):
            n = idx(i,j)
            coords[n] = (i*dx + (lp0 if (j%2)==1 else 0.0), j*dy)
    edges = []
    for j in range(ny):
        for i in range(nx):
            n = idx(i,j)
            if i<nx-1: edges.append((n, idx(i+1,j)))
            if j<ny-1:
                edges.append((n, idx(i,j+1)))
                if (j%2)==1 and i<nx-1:
                    edges.append((n, idx(i+1,j+1)))
                elif (j%2)==0 and i>0:
                    edges.append((n, idx(i-1,j+1)))
    edges = np.array(edges, int)
    elen = 2*lp0
    return coords, edges, elen

def build_square_lattice(L=2.0, W=2.0, lp0=lp0):
    """square lattice with spacing 2*lp0."""
    dx = 2*lp0; dy=dx
    nx = int(round(L/dx)+1)
    ny = int(round(W/dy)+1)
    N  = nx*ny
    coords = np.zeros((N,2), float)
    def idx(i,j): return j*nx+i
    for j in range(ny):
        for i in range(nx):
            coords[idx(i,j)] = (i*dx, j*dy)
    edges=[]
    for j in range(ny):
        for i in range(nx):
            n=idx(i,j)
            if i<nx-1: edges.append((n, idx(i+1,j)))
            if j<ny-1: edges.append((n, idx(i,j+1)))
    edges=np.array(edges,int)
    elen = 2*lp0
    return coords, edges, lp0, elen

# -------------------------- KCL assembly --------------------------

import numpy as np
from scipy import sparse

def edge_orientation(p0, p1):
    """
    Return one of {'horiz','vert','diag_br','diag_bl'} based on the closest
    of the four canonical axes.
    """
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 'horiz'
    vhat = v / n

    b_h = np.array([1.0, 0.0])                 # horizontal
    b_v = np.array([0.0, 1.0])                 # vertical
    b_br = np.array([0.5, -np.sqrt(3)/2.0])    # "\" TL→BR (tri lattice)
    b_bl = np.array([-0.5,  np.sqrt(3)/2.0])   # "/"  TR→BL

    bases = [b_h, b_v, b_br, b_bl]
    dots = [abs(np.dot(vhat, b)) for b in bases]
    return ['horiz', 'vert', 'diag_br', 'diag_bl'][int(np.argmax(dots))]

def assemble_kcl(coords, edges, speeds_mph=(20.0, 20.0, 25.0, 30.0), obstacle_fn=None, **kwargs):
    """
    Symmetric KCL (same speed both directions) with orientation-based speeds.
    speeds_mph = (horiz, vert, diag_br (\\), diag_bl (/))
    Returns:
      M, time_map, efrom, eto, etime, K_edge
    """
    mph_h, mph_v, mph_br, mph_bl = speeds_mph
    rows, cols, data = [], [], []
    efrom, eto, etime, K_edge = [], [], [], []
    time_map = {}

    for (u, v) in edges:
        p0, p1 = coords[u], coords[v]
        if obstacle_fn is not None and obstacle_fn(p0, p1):
            continue

        ori = edge_orientation(p0, p1)
        if ori == 'horiz':
            mph = mph_h
        elif ori == 'vert':
            mph = mph_v
        elif ori == 'diag_br':
            mph = mph_br
        else:
            mph = mph_bl

        speed = mph * MPH_TO_MPS
        elen  = np.linalg.norm(p1 - p0)
        t     = elen / max(speed, 1e-9)
        G     = 1.0  / max(t,     1e-12)

        # symmetric stamp
        rows += [u, v, u, v]
        cols += [u, v, v, u]
        data += [ G,  G, -G, -G]

        efrom.append(u); eto.append(v); etime.append(t); K_edge.append(G)
        time_map[(u, v)] = t
        time_map[(v, u)] = t

    N = coords.shape[0]
    M = sparse.csr_matrix((np.array(data, float),
                           (np.array(rows, int), np.array(cols, int))),
                          shape=(N, N))
    return M, time_map, np.array(efrom, int), np.array(eto, int), np.array(etime, float), np.array(K_edge, float)


def apply_dirichlet_row_replace(M, nodes, value):
    """Overwrite rows for Dirichlet nodes to phi=value."""
    ML = M.tolil(copy=True); 
    b = np.zeros(M.shape[0], float)
    for n in nodes:
        ML.rows[n] = [n]
        ML.data[n] = [1.0]
        b[n] = value
    return ML.tocsr(), b

def solve_with_component_pinning(M, b, pinned_nodes):
    """
    Solve robustly by pinning one node per disconnected component
    that lacks a Dirichlet pin. Removes nullspaces cleanly.
    """
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import spsolve

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


# ---------- sizing helpers: 15–20 in x, ~10 in y ----------
def lp0_for_square(L, W, nx_segments=18, ny_segments=10):
    # square grid has spacing dx = 2*lp0; nx-1 = L/dx ≈ nx_segments
    lp0_x = L / (2.0 * nx_segments)
    lp0_y = W / (2.0 * ny_segments)
    return min(lp0_x, lp0_y)

def lp0_for_hex(L, W, nx_segments=18, ny_segments=10):
    # hex grid: dx = 2*lp0; dy = sqrt(3)*lp0; target nx-1 ≈ nx_segments, ny-1 ≈ ny_segments
    lp0_x = L / (2.0 * nx_segments)
    lp0_y = W / (np.sqrt(3.0) * ny_segments)
    return min(lp0_x, lp0_y)

# -------------------------- path extraction --------------------------

def dijkstra_path(coords, efrom, eto, etime, src, dst):
    rows = np.concatenate([efrom, eto])
    cols = np.concatenate([eto, efrom])
    dat  = np.concatenate([etime, etime])
    N = coords.shape[0]
    adj = sparse.csr_matrix((dat,(rows,cols)), shape=(N,N))
    dist, preds = csgraph.dijkstra(adj, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]): return [], np.inf
    # backtrack
    path=[]; cur=dst
    while cur!=-9999 and cur!=src:
        path.append(cur); cur=preds[cur]
    path.append(src); path=path[::-1]
    return path, float(dist[dst])

def path_time(path, time_map):
    if len(path)<2: return np.nan
    tt=0.0
    for a,b in zip(path[:-1], path[1:]):
        t = time_map.get((a,b)) or time_map.get((b,a))
        if t is None: return np.nan
        tt+=t
    return tt


def path_overlap(a,b):
    def edgeset(p): return set((min(u,v),max(u,v)) for u,v in zip(p[:-1],p[1:]))
    Ea = edgeset(a); Eb = edgeset(b)
    if not Ea and not Eb: return 1.0
    if not Ea or not Eb:  return 0.0
    return len(Ea & Eb) / len(Ea | Eb)


def build_adjacency_list(coords, efrom, eto):
    """Build adjacency list from edge arrays."""
    N = coords.shape[0]
    adj = [[] for _ in range(N)]
    for u, v in zip(efrom, eto):
        adj[u].append(v)
        adj[v].append(u)
    return adj


def snap_continuous_path_to_network(path_xy, coords, efrom, eto, src, dst):
    """
    Snap a continuous gradient-traced path onto the physical network.

    Strategy:
    1. Sample points along the continuous path
    2. At each step from current node, pick the neighbor that best follows
       the continuous path direction toward the next sampled point
    3. Continue until reaching destination

    Args:
        path_xy: Nx2 array of continuous path coordinates
        coords: node coordinates (Mx2)
        efrom, eto: edge connectivity arrays
        src, dst: source and destination node indices

    Returns:
        node_path: list of node indices forming a valid network path
    """
    if len(path_xy) < 2:
        return [src, dst] if src != dst else [src]

    adj = build_adjacency_list(coords, efrom, eto)

    # Sample points along the continuous path at regular intervals
    # Use cumulative arc length to parameterize
    diffs = np.diff(path_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cumlen[-1]

    if total_len < 1e-12:
        return [src, dst] if src != dst else [src]

    # Create interpolator for the continuous path
    from scipy.interpolate import interp1d
    path_interp_x = interp1d(cumlen, path_xy[:, 0], kind='linear', fill_value='extrapolate')
    path_interp_y = interp1d(cumlen, path_xy[:, 1], kind='linear', fill_value='extrapolate')

    node_path = [src]
    current = src
    visited = {src}

    # Track position along the continuous path
    path_progress = 0.0

    max_steps = len(coords)  # prevent infinite loops
    step_count = 0

    while current != dst and step_count < max_steps:
        step_count += 1
        neighbors = adj[current]

        if not neighbors:
            break

        # Find where we are on the continuous path
        current_pos = coords[current]

        # Look ahead on continuous path to find target direction
        # Find the closest point on the continuous path to current position
        dists_to_path = np.linalg.norm(path_xy - current_pos, axis=1)
        closest_idx = np.argmin(dists_to_path)
        path_progress = cumlen[closest_idx]

        # Look ahead by approximately one edge length
        avg_edge_len = np.mean([np.linalg.norm(coords[u] - coords[v])
                                for u, v in zip(efrom[:100], eto[:100])])
        lookahead = min(path_progress + 2 * avg_edge_len, total_len)

        target_x = path_interp_x(lookahead)
        target_y = path_interp_y(lookahead)
        target_pos = np.array([target_x, target_y])

        # Direction we want to go
        desired_dir = target_pos - current_pos
        desired_norm = np.linalg.norm(desired_dir)
        if desired_norm > 1e-12:
            desired_dir = desired_dir / desired_norm

        # Score each neighbor: prefer direction alignment + progress toward dst
        best_neighbor = None
        best_score = -np.inf

        dst_pos = coords[dst]
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

            # Alignment with desired direction (from continuous path)
            alignment = np.dot(edge_dir, desired_dir)

            # Progress toward destination
            nb_dist_to_dst = np.linalg.norm(nb_pos - dst_pos)
            progress = current_dist_to_dst - nb_dist_to_dst

            # Combined score: favor alignment, with bonus for progress
            score = alignment + 0.3 * (progress / max(edge_len, 1e-12))

            # Big bonus if this is the destination
            if nb == dst:
                score += 10.0

            if score > best_score:
                best_score = score
                best_neighbor = nb

        if best_neighbor is None:
            # Stuck - try any unvisited neighbor
            for nb in neighbors:
                if nb not in visited:
                    best_neighbor = nb
                    break

        if best_neighbor is None:
            # Completely stuck
            break

        node_path.append(best_neighbor)
        visited.add(best_neighbor)
        current = best_neighbor

    # If we didn't reach dst, try to connect via shortest remaining path
    if current != dst:
        # Use BFS to find shortest path from current to dst
        from collections import deque
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
            node_path.extend(found_path[1:])  # skip current, already in path

    return node_path

# -------------------------- CASES --------------------------

def showconductance(coords,efrom,eto,M_jam,off,i,save_prefix,show):
    # Conductance visualization (no interpolation): scatter edge midpoints colored by G
    midpts = 0.5 * (coords[efrom] + coords[eto])
    conductance = np.array([-M_jam[u, v] for u, v in zip(efrom, eto)], dtype=float)
    print("minimum conductance : ", max(conductance))
    print("minimum conductance : ", min(conductance))
    plt.figure(figsize=(7.2, 5.6))
    plt.scatter(midpts[:,0], midpts[:,1], c=conductance, s=8, cmap='jet')
    plt.colorbar(label='Edge conductance (1/s)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Conductance after M-jam, offset {off:.2f}×lp0')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.colorbar()
    cond_png = f"{save_prefix}_{i}_conductivity.png"
    plt.tight_layout(); plt.savefig(cond_png, dpi=220)
    if not show:
            plt.close()
    print("Saved conductivity field:", cond_png)


def showbjmesh(coords,bj,off,i,save_prefix,show,title,figname):
    # Plot bj as a mesh over node coordinates
    plt.figure(figsize=(7.2, 5.6))
    bx = coords[:,0]; by = coords[:,1]
    try:
            pcm = plt.tricontourf(bx, by, bj, levels=20, cmap='cool')
            plt.colorbar(pcm, label='bj value')
    except Exception:
            plt.scatter(bx, by, c=bj, s=8, cmap='cool')
            plt.colorbar(label='bj value')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title(f'bj RHS after Dirichlet (offset {off:.2f}×lp0)')
    plt.title(title)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    #bj_png = f"{save_prefix}_{i}_bj.png"
    bj_png = f"{save_prefix}_{i}_{figname}"
    plt.tight_layout(); plt.savefig(bj_png, dpi=220)
    if not show:
            plt.close()
    print("Saved bj plot:", figname)


def showphimesh(coords,phi,off,i,save_prefix,show,title,figname):
    plt.figure(figsize=(7.2, 5.6))
    plt.scatter(coords[:,0], coords[:,1], c=phi, s=8, cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title(f'Phi field (offset {off:.2f}×lp0)')
    plt.title(title)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.colorbar()
    phi_png = f"{save_prefix}_{i}_{figname}"
    plt.tight_layout(); plt.savefig(phi_png, dpi=220)
    if not show:
        plt.close()
    print("Saved Phi field:", phi_png)

def greedy_path(phi,efrom,eto,src,dst,time_map_jam,coords,choice):
    greedy_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, choice=choice, coords=coords)
    greedy_time = path_time(greedy_nodes, time_map_jam)
    return greedy_nodes, greedy_time


# -------------------------- gradient tracing --------------------------


def gradienttrace(gradx_interp,grady_interp,coords,src,dst,steplength=1e-4,
                max_steps=7000, initial_angle_deg=4):
    """
    Follow continuous -grad(phi) by interpolating phi to a grid. Returns a snapped
    node path and the continuous polyline (Nx2).
    """
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    pt = coords[int(src)].astype(float)
    goal = coords[int(dst)].astype(float)
    path_xy = [pt.copy()]

    # First step: bias toward goal, optionally rotated
    if np.linalg.norm(goal - pt) > 1e-12:
        dir_goal = (goal - pt) / np.linalg.norm(goal - pt)
        if initial_angle_deg:
            ang = np.deg2rad(initial_angle_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            dir_goal = np.array([ca*dir_goal[0] - sa*dir_goal[1],
                                 sa*dir_goal[0] + ca*dir_goal[1]], float)
        gx0 = gradx_interp((pt[1], pt[0]))
        gy0 = grady_interp((pt[1], pt[0]))
        gnrm = np.sqrt(gx0*gx0+gy0*gy0)
        if np.isfinite(gx0) and np.isfinite(gy0) and gnrm>1e-6:
            gvec0 = np.array([gx0, gy0], float)
            gnorm0 = np.linalg.norm(gvec0)
            if gnorm0 > 1e-15:
                gdir0 = gvec0 / gnorm0
                alpha = 0.0  # favor goal direction for the first step
                step_dir = alpha * (-gdir0) + (1 - alpha) * dir_goal
                step_dir /= max(np.linalg.norm(step_dir), 1e-15)
            else:
                step_dir = dir_goal
        else:
            step_dir = dir_goal
        pt = pt + steplength * step_dir
        pt[0] = np.clip(pt[0], x_min, x_max)
        pt[1] = np.clip(pt[1], y_min, y_max)
        path_xy.append(pt.copy())

    for index in range(int(max_steps)):
        gx = gradx_interp((pt[1], pt[0]))
        gy = grady_interp((pt[1], pt[0]))
        if not np.isfinite(gx) or not np.isfinite(gy):
            print("non-finite gradient, stopping")
            break
        gvec = np.array([gx, gy], float)
        gnrm = np.linalg.norm(gvec)
        gn = 0*gvec
        if gnrm < 1e-6:
            dirgoal = (goal - pt) / np.linalg.norm(goal - pt)
            step = steplength*dirgoal
            print("near zero grad, heading to goal")
            exit()
        else:
            gn = gvec/gnrm
            step = -steplength * gn
        pt = pt + step
        pt[0] = np.clip(pt[0], x_min, x_max)
        pt[1] = np.clip(pt[1], y_min, y_max)
        path_xy.append(pt.copy())
        #print("index = ", index, "loc = ", pt," grd = ", gn,"gnrm =", gnrm)

        condition_to_goal = np.abs(pt[0]-goal[0])<=steplength and np.linalg.norm(pt-goal)<=1000*steplength
        if condition_to_goal:
            path_xy.append(goal.copy())
            break 
    return np.vstack(path_xy) if path_xy else np.zeros((0, 2))

def time_map_to_conductivity_mesh(time_map, coords, efrom, eto, xg, yg):
    """
    Convert edge-based time_map to a continuous conductivity mesh grid.
    
    Args:
        time_map: Dictionary mapping (u,v) edges to travel times
        coords: Node coordinates (Nx2)
        efrom, eto: Edge connectivity arrays
        xg: x-axis grid coordinates
        yg: y-axis grid coordinates
    
    Returns:
        conductivity_grid: 2D mesh of conductivity values (1/time)
    """
    # Extract edge midpoints and conductivity values
    edge_midpts = 0.5 * (coords[efrom] + coords[eto])
    edge_times = np.array([time_map[(u, v)] for u, v in zip(efrom, eto)])
    edge_len = np.array([np.linalg.norm(coords[u] - coords[v]) for u, v in zip(efrom, eto)])
    edge_conductivity = edge_len / np.maximum(edge_times, 1e-12)
    # Create mesh grid and interpolate
    X, Y = np.meshgrid(xg, yg)
    conductivity_grid = griddata(edge_midpts, edge_conductivity, (X, Y), method='nearest')
    # Smooth with NaN-aware blur
    #conductivity_grid = nan_gaussian_blur(conductivity_grid, 0.8)
    return conductivity_grid


def path_time_from_mesh(path_coords, conductivity_grid, xg, yg, steplength=1e-4):
    """
    Compute travel time along a path using interpolated conductivity from a mesh grid.
    Integrates in substeps of steplength to handle spatially-varying conductivity.
    
    Args:
        path_coords: Nx2 array of (x, y) coordinates defining the path
        conductivity_grid: 2D mesh grid of conductivity values (inverse of time per unit length)
        xg: x-axis grid coordinates
        yg: y-axis grid coordinates
        steplength: step size for integration along path segments
    
    Returns:
        Total travel time along the path (sum of segment times)
    """
    # from scipy.interpolate import RegularGridInterpolator
    
    if len(path_coords) < 2:
        print("path too short")
        return np.nan
    
    # Create interpolator for conductivity mesh
    # Note: conductivity_grid is indexed as [y, x] so we reverse xg, yg order
    cond_interp = RegularGridInterpolator((yg, xg), conductivity_grid, 
                                          bounds_error=False, fill_value=0.0)
    total_time = 0.0
    # Sum travel time over each path segment, integrating in steplength increments
    for i in range(len(path_coords) - 1):
        p_start = path_coords[i]
        p_end = path_coords[i + 1]
        segment_length = np.linalg.norm(p_end - p_start)
        if segment_length < 1e-12:
            continue
        # Number of substeps along this segment
        n_substeps = max(1, int(np.ceil(segment_length / steplength)))
        substep_length = segment_length / n_substeps
        # Integrate over substeps
        for j in range(n_substeps):
            # Position along segment (midpoint of substep)
            t_frac = (j + 0.5) / n_substeps
            pos = p_start + t_frac * (p_end - p_start)
            # Interpolate conductivity at this position
            # RegularGridInterpolator expects (y, x) order
            cond_value = cond_interp((pos[1], pos[0]))
            if not np.isfinite(cond_value) or cond_value <= 0:
                print(f"non-finite conductivity at segment {i}, substep {j}")
                return np.nan
            # Time = distance / speed, where speed ~ conductivity
            # Add time for this substep
            total_time += substep_length / cond_value
    return total_time


def case_hex_with_jam_offsets(save_prefix="jam_hex", show=False, jam_factor=1e7,offsets=[0.0, 10.0, 15.0, 20.0, 40.0]):
    """
    Two-node hex lattice with a circular jam (radius=lp0) sliding vertically from the
    midpoint between electrodes. Edges intersecting the disk have their travel time
    multiplied by jam_factor; plots greedy gradient vs Dijkstra paths for each offset.
    """
    R = lp0
    L, W = 2.0, np.sqrt(3.0)
    # lp0_used = lp0
    coords, edges, elen = build_tri_lattice(L=L, W=W, lp0=lp0)

    src_c = (0.30, W/2.0)
    dst_c = (L-0.30, W/2.0)
    M, time_map, efrom, eto, etime, K_edge = assemble_kcl(coords, edges, speeds_mph=(20, 20, 20, 20))

    # Force isotropic conductance to 60,000 S on every edge
    iso_G = 60000.0
    base_t = 1.0 / iso_G
    etime = np.full_like(etime, base_t, dtype=float)
    K_edge = np.full_like(K_edge, iso_G, dtype=float)
    time_map = {}
    for u, v in zip(efrom, eto):
        time_map[(u, v)] = base_t
        time_map[(v, u)] = base_t

    r_src = np.linalg.norm(coords - np.array(src_c)[None,:], axis=1)
    r_dst = np.linalg.norm(coords - np.array(dst_c)[None,:], axis=1)
    src_nodes = np.where(r_src<=R)[0]
    dst_nodes = np.where(r_dst<=R)[0]
    print("Source nodes:", src_nodes)
    print("Dest   nodes:", dst_nodes)
    print("efrom:", efrom)
    print("eto:", eto)

    ML, b  = apply_dirichlet_row_replace(M, src_nodes, 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, dst_nodes, 0.0)
    b += b2
    if len(src_nodes)==0:
        src_nodes = [nearest_node(coords, src_c)]
        ML = ML.tolil(); ML.rows[src_nodes[0]]=[src_nodes[0]]; ML.data[src_nodes[0]]=[1.0]; b[src_nodes[0]] = 1.0; ML=ML.tocsr()
    if len(dst_nodes)==0:
        dst_nodes = [nearest_node(coords, dst_c)]
        ML = ML.tolil(); ML.rows[dst_nodes[0]]=[dst_nodes[0]]; ML.data[dst_nodes[0]]=[1.0]; b[dst_nodes[0]] = 0.0; ML=ML.tocsr()

    src = nearest_node(coords, src_c); dst = nearest_node(coords, dst_c)
    mid = 0.5 * (coords[src] + coords[dst])
    ng = 260
    xg = np.linspace(0, L, ng); 
    yg = np.linspace(0, W, ng)
    X, Y = np.meshgrid(xg, yg)

    results = []
    for i, off in enumerate(offsets, start=1):
        center = mid + np.array([0.0, off*lp0])
        # print center x and y coordinates for debugging/inspection
        try:
            cx, cy = float(center[0]), float(center[1])
        except Exception:
            cx, cy = center[0], center[1]
        print(f"center: x={cx:.6f}, y={cy:.6f}")

        slow_mask_arr = np.array([seg_intersects_circle(coords[u], coords[v], center, RADIUS)
                                  for u, v in zip(efrom, eto)], dtype=bool)

        rows, cols, data = [], [], []
        etime_jam = etime.copy()
        time_map_jam = {}
        for idx, (u, v, t0) in enumerate(zip(efrom, eto, etime)):
            slow = slow_mask_arr[idx]
            t = t0 * jam_factor if slow else t0
            etime_jam[idx] = t
            G = 1.0 / max(t, 1e-12)
            rows += [u, v, u, v]; 
            cols += [u, v, v, u]; 
            data += [G, G, -G, -G]
            time_map_jam[(u, v)] = t; 
            time_map_jam[(v, u)] = t
        M_jam = sparse.csr_matrix((np.array(data, float),
                                   (np.array(rows, int), np.array(cols, int))),
                                   shape=(coords.shape[0], coords.shape[0]))

        ML_j, bj  = apply_dirichlet_row_replace(M_jam, src_nodes, 100)
        ML_j, b2j = apply_dirichlet_row_replace(ML_j, dst_nodes, -100)
        bj += b2j

        phi = solve_with_component_pinning(ML_j, bj, pinned_nodes=np.r_[src_nodes, dst_nodes])
        Phi_lin = griddata(coords, phi, (X, Y), method='linear')
        Phi_nn  = griddata(coords, phi, (X, Y), method='nearest')
        Phi = np.where(np.isfinite(Phi_lin), Phi_lin, Phi_nn)  # fill boundary holes so the field reaches x=2
        #Phi = nan_gaussian_blur(Phi, 0.8)
        dPhidy, dPhidx = np.gradient(Phi, yg[1]-yg[0], xg[1]-xg[0], edge_order=2)

        # Potential visualization (no interpolation): scatter nodes colored by phi
        # r = ML_j.dot(phi)
        # idx = 20189
        showphimesh(coords,phi,off,i,save_prefix,show,"phi value","phi")
        # row = (ML_j.getrow(idx)).toarray()
        # showphimesh(coords,row,off,i,save_prefix,show,"ML_j value","MLj")
        # showbjmesh(coords,bj,off,i,save_prefix,show,"bj value","bj")
        # showbjmesh(coords,r,off,i,save_prefix,show,"r value","r")
        # showbjmesh(coords,r-bj,off,i,save_prefix,show,"Aphi value","aphi")
        # showconductance(coords,efrom,eto,M_jam,off,i,save_prefix,show)
        
        dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime_jam, src, dst)

        gradx_interp = RegularGridInterpolator((yg, xg), dPhidx, method='quintic', bounds_error=False, fill_value=0.0)
        grady_interp = RegularGridInterpolator((yg, xg), dPhidy, method='quintic', bounds_error=False, fill_value=0.0)
        conductivity_grid = time_map_to_conductivity_mesh(time_map, coords, efrom, eto, xg, yg)
        greedy_xy = gradienttrace(gradx_interp,grady_interp, coords, src, dst, steplength=steplen)
        greedy_time = path_time_from_mesh(greedy_xy, conductivity_grid, xg, yg, steplength=steplen)

        # Snap the continuous gradient path onto the physical network
        snapped_nodes = snap_continuous_path_to_network(greedy_xy, coords, efrom, eto, src, dst)
        snapped_time = path_time(snapped_nodes, time_map_jam)

        # Paths overlay on scatter (no interpolation)
        plt.figure(figsize=(7.2, 5.6))
        plt.scatter(coords[:,0], coords[:,1], c=phi, s=6, cmap='viridis')

        # Plot snapped network path (follows actual edges)
        if snapped_nodes:
            plt.plot(coords[snapped_nodes,0], coords[snapped_nodes,1], 'r-', lw=1.5, label='Gradient')

        if dijk_nodes:
            plt.plot(coords[dijk_nodes,0], coords[dijk_nodes,1], 'm-', lw=1.0, label='Dijkstra')

        try:
            CS = plt.contour(X, Y, Phi, levels=15, colors='k', linewidths=0.6, alpha=0.6)
            plt.clabel(CS, inline=True, fontsize=7, fmt='%.2f')
        except Exception:
            pass
        
        # Add arrows for steepest descent of phi (-grad phi) from gridded field
        # stride = 2
        # Xs = X[::stride, ::stride]; Ys = Y[::stride, ::stride]
        # U = -dPhidx[::stride, ::stride]; V = -dPhidy[::stride, ::stride]
        # mask = np.isfinite(Phi[::stride, ::stride]) & np.isfinite(U) & np.isfinite(V)

        # plt.quiver(Xs[mask], Ys[mask], U[mask], V[mask],
        #            color='white', angles='xy', scale_units='xy',
        #            scale=2000, width=0.0015, alpha=0.5, minlength=0.05)

        ax = plt.gca()
        plt.plot(coords[src,0], coords[src,1], 'go', ms=1, label='Source')
        plt.plot(coords[dst,0], coords[dst,1], 'ro', ms=1, label='Destination')
        ax.add_patch(Circle((center[0], center[1]), RADIUS, fill=False, ec='orange', lw=1.0, label='Jam'))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, L); ax.set_ylim(0, W)
        ax.margins(x=0.0, y=0.0)  # remove auto-padding so the color fills to the boundary

        # Annotate travel times for each path directly on the plot
        def _fmt_time(label, val):
            return f"{label}: {val:.4f} s" if np.isfinite(val) else f"{label}: inf"
        time_text = _fmt_time("Dijkstra", dijk_time) + "\n" + _fmt_time("Gradient", snapped_time)
        plt.gca().text(
            0.02, 0.98, time_text,
            transform=plt.gca().transAxes,
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3),
            fontsize=7,
        )

        plt.title(f'Jam offset {off:.2f}×lp0')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.legend(loc='lower right')
        png_path = f"{save_prefix}_{i}.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=220, bbox_inches='tight', pad_inches=0.06)

        if not show:
            plt.close('all')
        
        results.append(dict(
            case=i,
            offset=float(off),
            center=(float(center[0]), float(center[1])),
            slowed_edges=int(sum(slow_mask_arr)),
            greedy_time_continuous=float(greedy_time),
            greedy_time_snapped=float(snapped_time),
            dijkstra_time=float(dijk_time),
            png=png_path,
        ))
        print(f"dijkstra={dijk_time:.5f}s gradient={snapped_time:.5f}s -> {png_path}")

    if show:
        plt.show()
    return results

if __name__ == "__main__":
    case_hex_with_jam_offsets(show=False)
    plt.show()
