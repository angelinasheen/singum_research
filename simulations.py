
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
from scipy.sparse import csgraph
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle


MPH_TO_MPS = 0.44704

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

def build_adjacency(edges_from, edges_to, N):
    from collections import defaultdict
    nbrs = defaultdict(list)
    for u,v in zip(edges_from, edges_to):
        nbrs[u].append(v); nbrs[v].append(u)
    return nbrs

# -------------------------- geometry tests for obstacles --------------------------

def seg_intersects_rect(p0, p1, cx, cy, w, h):
    xmin, xmax = cx - w/2, cx + w/2
    ymin, ymax = cy - h/2, cy + h/2
    x0,y0 = p0; x1,y1 = p1
    if max(x0,x1)<xmin or min(x0,x1)>xmax or max(y0,y1)<ymin or min(y0,y1)>ymax:
        return False
    dx, dy = x1-x0, y1-y0
    if dx==0 and dy==0:
        return (xmin<=x0<=xmax) and (ymin<=y0<=ymax)
    # project rect center to segment and test inside rect
    t = ((cx-x0)*dx + (cy-y0)*dy)/(dx*dx+dy*dy)
    t = np.clip(t, 0.0, 1.0)
    xc, yc = x0 + t*dx, y0 + t*dy
    return (xmin<=xc<=xmax) and (ymin<=yc<=ymax)

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

def build_tri_lattice(L=2.0, W=np.sqrt(3.0), lp0=0.01):
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
    return coords, edges, lp0, elen

def build_square_lattice(L=2.0, W=2.0, lp0=0.01):
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
    ML = M.tolil(copy=True); b = np.zeros(M.shape[0], float)
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

#singum tensors and flows

def singum_tensor_per_node(coords, efrom, eto, K_edge, lattice, lp0):
    """
    Compute per-node singum mobility tensor k = (2 lp^2 / V_s) sum_e K (n n^T),
    where lp = edge_length/2, V_s hex = 2*sqrt(3)*lp0^2, square = 4*lp0^2.
    """
    N = coords.shape[0]
    kxx = np.zeros(N); kxy = np.zeros(N); kyy = np.zeros(N)
    V_s = 2*np.sqrt(3)*lp0**2 if lattice=='hex' else 4*lp0**2

    for u,v,K in zip(efrom, eto, K_edge):
        d = coords[v]-coords[u]
        elen = np.linalg.norm(d)
        if elen<=0: continue
        nvec = d/elen
        lp = 0.5*elen
        factor = (2*lp*lp*K)/V_s  # derived from paper
        # contribute to node u (outward direction from u to v)
        kxx[u] += factor * (nvec[0]*nvec[0])
        kxy[u] += factor * (nvec[0]*nvec[1])
        kyy[u] += factor * (nvec[1]*nvec[1])
        # and node v (outward from v to u is -n)
        kxx[v] += factor * (nvec[0]*nvec[0])
        kxy[v] += factor * (nvec[0]*nvec[1])
        kyy[v] += factor * (nvec[1]*nvec[1])

    return kxx, kxy, kyy

def singum_flow_per_node(coords, phi, efrom, eto, time_map, lattice, lp0):
    """
    q = (1/V_s) sum_e x^I Q_e, with x^I = lp n (vector from node to cutpoint),
    Q_e = G_e (phi_u - phi_v), G_e = 1/t_e.
    """
    N = coords.shape[0]
    qx = np.zeros(N); qy = np.zeros(N)
    V_s = 2*np.sqrt(3)*lp0**2 if lattice=='hex' else 4*lp0**2

    for u,v in zip(efrom, eto):
        p0, p1 = coords[u], coords[v]
        d = p1-p0
        elen = np.linalg.norm(d)
        if elen<=0: continue
        nvec = d/elen
        lp = 0.5*elen
        G = 1.0 / time_map[(u,v)]
        Iuv = G*(phi[u]-phi[v])  # flow from u to v positive if phi[u]>phi[v]
        # contribution to node u: outward direction +n
        xI = lp*nvec
        qx[u] += xI[0]*Iuv
        qy[u] += xI[1]*Iuv
        # contribution to node v: outward direction -n, flow from v to u is -Iuv
        # So xI*(-Iuv) with xI = lp*(-n) -> identical sign as u side:
        qx[v] += xI[0]*Iuv
        qy[v] += xI[1]*Iuv

    qx /= V_s; qy /= V_s
    return qx, qy


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

# ---------- total-flow scaling (enforce 1000 vehicles) ----------
def total_outflow_from_set(phi, coords, efrom, eto, time_map, node_set):
    node_set = set(map(int, np.asarray(node_set).ravel()))
    Q = 0.0
    for u, v in zip(efrom, eto):
        if (u in node_set) ^ (v in node_set):  # exactly one in the set
            t = time_map[(u, v)]
            G = 1.0 / max(t, 1e-12)
            Q += G * (phi[u] - phi[v]) if (u in node_set) else G * (phi[v] - phi[u])
    return Q  # sign = out of the set

def scale_phi_to_total_flow(phi, coords, efrom, eto, time_map, node_set, Q_target):
    Q0 = total_outflow_from_set(phi, coords, efrom, eto, time_map, node_set)
    if abs(Q0) < 1e-18:
        return phi, 0.0
    alpha = Q_target / Q0
    return phi * alpha, alpha

# ---------- plot K_ij (mobility) ----------
def plot_mobility_tensors(coords, kxx, kxy, kyy, L, W, title_prefix='Mobility'):
    # Interpolate to a grid for clean panels
    ngx, ngy = 180, 140
    xg = np.linspace(0, L, ngx); yg = np.linspace(0, W, ngy)
    X, Y = np.meshgrid(xg, yg)
    Kxx = griddata(coords, kxx, (X, Y), method='linear')
    Kxy = griddata(coords, kxy, (X, Y), method='linear')
    Kyy = griddata(coords, kyy, (X, Y), method='linear')

    # Simple NaN-safe blur
    Kxx = nan_gaussian_blur(Kxx, 0.8)
    Kxy = nan_gaussian_blur(Kxy, 0.8)
    Kyy = nan_gaussian_blur(Kyy, 0.8)

    for Z, lab in [(Kxx, r'$K_{xx}$'), (Kxy, r'$K_{xy}$'), (Kyy, r'$K_{yy}$')]:
        plt.figure(figsize=(6.6, 4.8))
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        plt.colorbar(label=lab)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title(f'{title_prefix}: {lab}')
        plt.tight_layout()

# -------------------------- path extraction --------------------------

def greedy_path_on_nodes(phi, efrom, eto, src, dst=None, goal_set=None, coords=None, max_steps=500000):
    """
    Greedy steepest-drop path on node potentials with robust handling of Dirichlet plateaus.
    - src: int or iterable of start nodes (e.g., all nodes in a source disk)
    - dst: optional single goal node
    - goal_set: optional set of goal nodes (e.g., all nodes in a sink disk). If provided, overrides dst.
    - coords: optional (N,2) array for distance-based tie-breaking / plateau creep
    """
    from collections import defaultdict, deque
    nbrs = defaultdict(list)
    for u, v in zip(efrom, eto):
        nbrs[u].append(v); nbrs[v].append(u)

    N = len(phi)
    tol = 1e-12

    # Normalize inputs
    if goal_set is None:
        goal_set = set()
        if dst is not None:
            goal_set.add(int(dst))

    # Choose a start node.
    def pick_start_from_set(S):
        S = list(S)
        # 1) Prefer boundary nodes: have at least one neighbor with strictly lower phi
        boundary = [u for u in S if any(phi[v] < phi[u] - tol for v in nbrs[u])]
        if boundary:
            if coords is not None and goal_set:
                # choose boundary node closest to any goal (approx: closest to the first goal)
                g = next(iter(goal_set))
                return min(boundary, key=lambda k: np.linalg.norm(coords[k] - coords[g]))
            return boundary[0]
        # 2) Plateau crawl: BFS within equal-phi set to find a node touching lower phi
        seen = set(S)
        dq = deque(S)
        while dq:
            u = dq.popleft()
            if any(phi[v] < phi[u] - tol for v in nbrs[u]):
                return u
            for v in nbrs[u]:
                if v not in seen and abs(phi[v] - phi[u]) <= tol:
                    seen.add(v); dq.append(v)
        # 3) Fallback: just pick the member of S closest to the goal (or the first)
        if coords is not None and goal_set:
            g = next(iter(goal_set))
            return min(S, key=lambda k: np.linalg.norm(coords[k] - coords[g]))
        return S[0] if S else None

    if isinstance(src, (list, tuple, set, np.ndarray)):
        cur = pick_start_from_set(src)
    else:
        cur = int(src)

    if cur is None:
        return []

    # Walk
    path = [cur]
    visited = {cur}

    def reached_goal(u):
        return (u in goal_set) if goal_set else (u == dst)

    # distance helper for tie-breaking
    def dist_to_goal(u):
        if coords is None or not goal_set:
            return 0.0
        # nearest goal
        return min(np.linalg.norm(coords[u] - coords[g]) for g in goal_set)

    steps = 0
    while not reached_goal(cur) and steps < max_steps:
        steps += 1
        nlist = nbrs[cur]
        # Strictly descending neighbors
        lower = [v for v in nlist if phi[v] < phi[cur] - tol]
        if lower:
            # pick with minimum phi; tie-break by distance to goal if available
            best_phi = min(phi[v] for v in lower)
            candidates = [v for v in lower if abs(phi[v] - best_phi) <= tol]
            if len(candidates) > 1:
                nxt = min(candidates, key=dist_to_goal)
            else:
                nxt = candidates[0]
        else:
            # Plateau creep: move to equal-phi neighbor that gets closer to goal
            equal = [v for v in nlist if abs(phi[v] - phi[cur]) <= tol]
            if not equal:
                break
            # choose the equal-phi neighbor that minimizes distance to goal (if available)
            nxt = min(equal, key=dist_to_goal)

            # avoid trivial back-and-forth
            if len(path) >= 2 and nxt == path[-2]:
                # pick next best equal neighbor if possible
                eq_sorted = sorted(equal, key=dist_to_goal)
                for cand in eq_sorted:
                    if cand != path[-2]:
                        nxt = cand; break

        if nxt in visited and nxt != path[-2] if len(path) >= 2 else False:
            # We're looping on a plateau: stop
            break

        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return path
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

# -------------------------- plotting --------------------------

def plot_field_with_paths(X,Y,Phi,dPhidx,dPhidy, coords, src, dst,
                          greedy_coords=None, dijk_coords=None,
                          electrodes=None, hole_rect=None, crack_seg=None, title=''):
    plt.figure(figsize=(7.5,6))
    im = plt.pcolormesh(X, Y, Phi, shading='auto', cmap='viridis')
    cbar = plt.colorbar(im, label='Potential V (arb.)')
    # add smooth equipotential contours
    try:
        CS = plt.contour(X, Y, Phi, levels=12, colors='k', linewidths=0.8, alpha=0.55)
        # optional labels on a few lines:
        # plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f')
    except Exception:
        pass

    # (arrows are intentionally omitted)
    if greedy_coords is not None and len(greedy_coords)>0:
        plt.plot(greedy_coords[:,0], greedy_coords[:,1], 'c-', lw=3, label='Greedy steepest drop')
    if dijk_coords is not None and len(dijk_coords)>0:
        plt.plot(dijk_coords[:,0], dijk_coords[:,1], 'm-', lw=3, label='Dijkstra shortest time')
    plt.plot(coords[src,0], coords[src,1], 'go', ms=7, label='Source')
    plt.plot(coords[dst,0], coords[dst,1], 'ro', ms=7, label='Destination')
    ax = plt.gca()
    if electrodes:
        for (cx,cy,R,lab,col) in electrodes:
            ax.add_patch(Circle((cx,cy), R, fill=False, ec=col, lw=2.0, label=lab))
    if hole_rect is not None:
        cx,cy,w,h = hole_rect
        ax.add_patch(Rectangle((cx-w/2, cy-h/2), w,h, fill=False, ec='w', lw=2.0, ls='--', label='Hole'))
    if crack_seg is not None:
        a,b = np.asarray(crack_seg[0]).ravel()[:2], np.asarray(crack_seg[1]).ravel()[:2]
        plt.plot([a[0],b[0]],[a[1],b[1]], 'w--', lw=2.5, label='Crack')
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title(title)
    plt.legend(loc='best'); plt.tight_layout()

def plot_graph_and_path(coords, efrom, eto, path_nodes, title=''):
    segs = np.stack([coords[efrom], coords[eto]], axis=1)
    lc = LineCollection(segs, colors=(0.85,0.85,0.85,1.0), linewidths=0.3)
    plt.figure(figsize=(7.5,6)); ax=plt.gca()
    ax.add_collection(lc); ax.autoscale()
    if path_nodes and len(path_nodes)>0:
        c = coords[path_nodes]
        plt.plot(c[:,0], c[:,1], 'm-', lw=3, label='Dijkstra shortest time')
    ax.set_aspect('equal', adjustable='box')
    plt.title(title); plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.legend(loc='best'); plt.tight_layout()

# -------------------------- CASES --------------------------

def case_hex_with_disks(save_prefix=None, show=False, seed=42):

    # Grid size: ~18 across x, ~10 across y
    L, W = 2.0, np.sqrt(3.0)
    lp0 = lp0_for_hex(L, W, nx_segments=18, ny_segments=10)
    coords, edges, lp0, elen = build_tri_lattice(L=L, W=W, lp0=lp0)

    # Electrodes (disks) centered on midline
    R = 0.05
    src_c = (0.30, W/2.0); dst_c = (L-0.30, W/2.0)

    # Assemble (fix typo speeds_mph)
    M, time_map, efrom, eto, etime, K_edge = assemble_kcl(
        coords, edges, speeds_mph=(20,20,25,30)
    )

    # Dirichlet: pin all nodes inside each disk
    r_src = np.linalg.norm(coords - np.array(src_c)[None,:], axis=1)
    r_dst = np.linalg.norm(coords - np.array(dst_c)[None,:], axis=1)
    src_nodes = np.where(r_src<=R)[0]; dst_nodes = np.where(r_dst<=R)[0]
    ML, b  = apply_dirichlet_row_replace(M, src_nodes, 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, dst_nodes, 0.0)
    b += b2
    if len(src_nodes)==0:
        src_nodes = [nearest_node(coords, src_c)]
        ML = ML.tolil(); ML.rows[src_nodes[0]]=[src_nodes[0]]; ML.data[src_nodes[0]]=[1.0]; b[src_nodes[0]] = 1.0; ML=ML.tocsr()
    if len(dst_nodes)==0:
        dst_nodes = [nearest_node(coords, dst_c)]
        ML = ML.tolil(); ML.rows[dst_nodes[0]]=[dst_nodes[0]]; ML.data[dst_nodes[0]]=[1.0]; b[dst_nodes[0]] = 0.0; ML=ML.tocsr()

    phi = solve_with_component_pinning(ML, b, pinned_nodes=np.r_[src_nodes, dst_nodes])

    # --- Scale so total flow from the source disk equals 1000 vehicles ---
    phi, alpha = scale_phi_to_total_flow(phi, coords, efrom, eto, time_map, src_nodes, Q_target=1000.0)

    # Singum mobility tensor & flow (hex)
    kxx, kxy, kyy = singum_tensor_per_node(coords, efrom, eto, K_edge, lattice='hex', lp0=lp0)
    qx_node, qy_node = singum_flow_per_node(coords, phi, efrom, eto, time_map, lattice='hex', lp0=lp0)

    # Interpolate for viz
    ng = 260
    xg = np.linspace(0,L,ng); yg = np.linspace(0,W,ng)
    X,Y = np.meshgrid(xg,yg)
    Phi = griddata(coords, phi, (X,Y), method='linear')
    Phi = nan_gaussian_blur(Phi, 0.8)
    dPhidy, dPhidx = np.gradient(Phi, yg[1]-yg[0], xg[1]-xg[0], edge_order=2)

    # Paths (node potentials for greedy; Dijkstra on times)
    src = nearest_node(coords, src_c); dst = nearest_node(coords, dst_c)
    greedy_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)
    greedy_time = path_time(greedy_nodes, time_map); ov = path_overlap(greedy_nodes, dijk_nodes)

    title = f'Hexagonal lattice with source and sink'
    plot_field_with_paths(X,Y,Phi,dPhidx,dPhidy, coords, src, dst,
                          greedy_coords=coords[greedy_nodes] if greedy_nodes else None,
                          dijk_coords=coords[dijk_nodes] if dijk_nodes else None,
                          title=title)


    if save_prefix:
        plt.savefig(f'{save_prefix}_field.png', dpi=220)
        plt.figure(2); plt.savefig(f'{save_prefix}_graph.png', dpi=220)

    if show: plt.show()
    return dict(greedy_time=greedy_time, dijkstra_time=dijk_time, overlap=ov, alpha=alpha)


def case_hex_with_jam_offsets(save_prefix="jam_hex", show=False, jam_factor=100.0,
                              offsets=(0.0, 1/3, 2/3, 1.0, 2.0)):
    """
    Two-node hex lattice with a circular jam (radius=lp0) sliding vertically from the
    midpoint between electrodes. Edges intersecting the disk have their travel time
    multiplied by jam_factor; plots greedy gradient vs Dijkstra paths for each offset.
    """
    L, W = 2.0, np.sqrt(3.0)
    lp0 = lp0_for_hex(L, W, nx_segments=18, ny_segments=10)
    coords, edges, lp0, elen = build_tri_lattice(L=L, W=W, lp0=lp0)

    R = 0.05
    src_c = (0.30, W/2.0); dst_c = (L-0.30, W/2.0)

    M, time_map, efrom, eto, etime, K_edge = assemble_kcl(
        coords, edges, speeds_mph=(20, 20, 25, 30)
    )

    r_src = np.linalg.norm(coords - np.array(src_c)[None,:], axis=1)
    r_dst = np.linalg.norm(coords - np.array(dst_c)[None,:], axis=1)
    src_nodes = np.where(r_src<=R)[0]; dst_nodes = np.where(r_dst<=R)[0]
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
    xg = np.linspace(0, L, ng); yg = np.linspace(0, W, ng)
    X, Y = np.meshgrid(xg, yg)

    results = []
    for i, off in enumerate(offsets, start=1):
        center = mid + np.array([0.0, off*lp0])
        slow_mask = [seg_intersects_circle(coords[u], coords[v], center, lp0) for u, v in zip(efrom, eto)]

        rows, cols, data = [], [], []
        etime_jam = etime.copy()
        time_map_jam = {}
        for idx, (u, v, t0) in enumerate(zip(efrom, eto, etime)):
            slow = slow_mask[idx]
            t = t0 * jam_factor if slow else t0
            etime_jam[idx] = t
            G = 1.0 / max(t, 1e-12)
            rows += [u, v, u, v]; cols += [u, v, v, u]; data += [G, G, -G, -G]
            time_map_jam[(u, v)] = t; time_map_jam[(v, u)] = t
        M_jam = sparse.csr_matrix((np.array(data, float),
                                   (np.array(rows, int), np.array(cols, int))),
                                  shape=(coords.shape[0], coords.shape[0]))

        ML_j, bj  = apply_dirichlet_row_replace(M_jam, src_nodes, 1.0)
        ML_j, b2j = apply_dirichlet_row_replace(ML_j, dst_nodes, 0.0)
        bj += b2j
        phi = solve_with_component_pinning(ML_j, bj, pinned_nodes=np.r_[src_nodes, dst_nodes])

        Phi = griddata(coords, phi, (X, Y), method='linear')
        Phi = nan_gaussian_blur(Phi, 0.8)
        dPhidy, dPhidx = np.gradient(Phi, yg[1]-yg[0], xg[1]-xg[0], edge_order=2)

        greedy_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
        dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime_jam, src, dst)
        greedy_time = path_time(greedy_nodes, time_map_jam)

        title = f'Jam offset {off:.2f}×lp0 (factor×{jam_factor})'
        plot_field_with_paths(
            X, Y, Phi, dPhidx, dPhidy, coords, src, dst,
            greedy_coords=coords[greedy_nodes] if greedy_nodes else None,
            dijk_coords=coords[dijk_nodes] if dijk_nodes else None,
            electrodes=[(center[0], center[1], lp0, 'Jam', 'orange')],
            title=title
        )
        png_path = f"{save_prefix}_{i}.png"
        plt.savefig(png_path, dpi=220)
        if not show:
            plt.close('all')

        results.append(dict(
            case=i,
            offset=float(off),
            center=(float(center[0]), float(center[1])),
            slowed_edges=int(sum(slow_mask)),
            greedy_time=float(greedy_time),
            dijkstra_time=float(dijk_time),
            png=png_path,
        ))
        print(f"[Jam case {i}] offset={off:.3f}×lp0 slowed={sum(slow_mask)} edges "
              f"greedy={greedy_time:.3f}s dijkstra={dijk_time:.3f}s -> {png_path}")

    if show:
        plt.show()
    return results


def case_square_with_hole(save_prefix=None, show=False, seed=44):
    L = W = 2.0
    lp0 = lp0_for_square(L, W, nx_segments=18, ny_segments=10)
    coords, edges, lp0, elen = build_square_lattice(L=L, W=W, lp0=lp0)

    cx, cy = L/2, W/2
    hole_w, hole_h = 0.40, 0.40
    def obstacle_fn(p0, p1): 
        return seg_intersects_rect(p0, p1, cx, cy, hole_w, hole_h)

    M, time_map, efrom, eto, etime, K_edge = assemble_kcl(
        coords, edges, speeds_mph=(20.0, 20.0, 25.0, 30.0), obstacle_fn=obstacle_fn
    )

    tol = 1e-9
    left_nodes  = np.where(np.abs(coords[:,0] - 0.0) <= tol)[0]
    right_nodes = np.where(np.abs(coords[:,0] - L  ) <= tol)[0]
    ML, b  = apply_dirichlet_row_replace(M, left_nodes, 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, right_nodes, 0.0)
    b += b2

    phi = solve_with_component_pinning(ML, b, pinned_nodes=np.r_[left_nodes, right_nodes])

    # Scale to Q_total = 1000 vehicles through the left plate
    phi, alpha = scale_phi_to_total_flow(phi, coords, efrom, eto, time_map, left_nodes, Q_target=1000.0)

    kxx, kxy, kyy = singum_tensor_per_node(coords, efrom, eto, K_edge, lattice='square', lp0=lp0)
    qx_node, qy_node = singum_flow_per_node(coords, phi, efrom, eto, time_map, lattice='square', lp0=lp0)

    ng = 260
    xg = np.linspace(0, L, ng); yg = np.linspace(0, W, ng)
    X, Y = np.meshgrid(xg, yg)
    Phi = griddata(coords, phi, (X, Y), method='linear')
    hole_mask = (X >= cx - hole_w/2) & (X <= cx + hole_w/2) & (Y >= cy - hole_h/2) & (Y <= cy + hole_h/2)
    Phi[hole_mask] = np.nan
    Phi = nan_gaussian_blur(Phi, 0.8)
    dPhidy, dPhidx = np.gradient(Phi, yg[1]-yg[0], xg[1]-xg[0], edge_order=2)

    src = nearest_node(coords, (0.0, W/2))
    dst = nearest_node(coords, (L,   W/2))
    greedy_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)
    greedy_time = path_time(greedy_nodes, time_map); ov = path_overlap(greedy_nodes, dijk_nodes)

    title = f'Square lattice with hole'
    plot_field_with_paths(
        X, Y, Phi, dPhidx, dPhidy, coords, src, dst,
        greedy_coords=coords[greedy_nodes] if greedy_nodes else None,
        dijk_coords=coords[dijk_nodes] if dijk_nodes else None,
        hole_rect=(cx, cy, hole_w, hole_h),
        title=title
    )

    if save_prefix:
        plt.savefig(f'{save_prefix}_field.png', dpi=220)
        plt.figure(2); plt.savefig(f'{save_prefix}_graph.png', dpi=220)
    if show:
        plt.show()

    return dict(greedy_time=greedy_time, dijkstra_time=dijk_time, overlap=ov, alpha=alpha)

def case_square_with_vertical_conductive_strip(save_prefix=None, show=False, seed=123):
    L = W = 2.0
    lp0 = lp0_for_square(L, W, nx_segments=18, ny_segments=10)
    coords, edges, lp0, elen = build_square_lattice(L=L, W=W, lp0=lp0)

    strip_xc   = L/2
    strip_w    = 0.10
    strip_y0   = 0.60   # bottom
    strip_y1   = 1.40   # top
    gain_gamma = 4.0

    def in_strip(mx, my):
        return (abs(mx - strip_xc) <= strip_w/2) and (strip_y0 <= my <= strip_y1)

    mph_h, mph_v, mph_br, mph_bl = (20.0, 20.0, 25.0, 30.0)
    rows, cols, data = [], [], []
    efrom, eto, etime, K_edge = [], [], [], []
    time_map = {}

    for (u, v) in edges:
        p0, p1 = coords[u], coords[v]
        ori = edge_orientation(p0, p1)
        mph = mph_h if ori == 'horiz' else mph_v if ori == 'vert' else mph_br if ori == 'diag_br' else mph_bl
        speed = mph * MPH_TO_MPS
        el = np.linalg.norm(p1 - p0)
        t0 = el / max(speed, 1e-9)
        G0 = 1.0 / max(t0, 1e-12)

        mid = 0.5 * (p0 + p1)
        gain = gain_gamma if in_strip(mid[0], mid[1]) else 1.0
        G = G0 * gain; t = t0 / gain

        rows += [u, v, u, v]; cols += [u, v, v, u]; data += [G, G, -G, -G]
        efrom.append(u); eto.append(v); etime.append(t); K_edge.append(G)
        time_map[(u, v)] = t; time_map[(v, u)] = t

    M = sparse.csr_matrix((np.array(data, float),
                           (np.array(rows, int), np.array(cols, int))),
                          shape=(coords.shape[0], coords.shape[0]))
    efrom = np.array(efrom, int); eto = np.array(eto, int)
    etime = np.array(etime, float); K_edge = np.array(K_edge, float)

    tol = 1e-12
    left_nodes  = np.where(np.abs(coords[:, 0] - 0.0) <= tol)[0]
    right_nodes = np.where(np.abs(coords[:, 0] - L  ) <= tol)[0]
    ML, b  = apply_dirichlet_row_replace(M, left_nodes, 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, right_nodes, 0.0)
    b += b2

    phi = solve_with_component_pinning(ML, b, pinned_nodes=np.r_[left_nodes, right_nodes])

    # Scale to Q_total = 1000 vehicles through the left plate
    phi, alpha = scale_phi_to_total_flow(phi, coords, efrom, eto, time_map, left_nodes, Q_target=1000.0)

    kxx, kxy, kyy = singum_tensor_per_node(coords, efrom, eto, K_edge, lattice='square', lp0=lp0)
    qx_node, qy_node = singum_flow_per_node(coords, phi, efrom, eto, time_map, lattice='square', lp0=lp0)

    ng = 260
    xg = np.linspace(0, L, ng); yg = np.linspace(0, W, ng)
    X, Y = np.meshgrid(xg, yg)
    Phi = griddata(coords, phi, (X, Y), method='linear')
    Phi = nan_gaussian_blur(Phi, 0.8)
    dPhidy, dPhidx = np.gradient(Phi, yg[1]-yg[0], xg[1]-xg[0], edge_order=2)

    src = nearest_node(coords, (0.0, W/2))
    dst = nearest_node(coords, (L,   W/2))
    greedy_nodes          = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)
    greedy_time           = path_time(greedy_nodes, time_map)
    ov                    = path_overlap(greedy_nodes, dijk_nodes)

    title = f'Square lattice with vertical conductive strip'
    hole_rect = (strip_xc, 0.5*(strip_y0+strip_y1), strip_w, (strip_y1-strip_y0))  # visual only

    plot_field_with_paths(
        X, Y, Phi, dPhidx, dPhidy, coords, src, dst,
        greedy_coords=coords[greedy_nodes] if greedy_nodes else None,
        dijk_coords=coords[dijk_nodes] if dijk_nodes else None,
        electrodes=None,
        hole_rect=hole_rect,
        title=title
    )

    if save_prefix:
        plt.savefig(f'{save_prefix}_field.png', dpi=220)
        plt.figure(2); plt.savefig(f'{save_prefix}_graph.png', dpi=220)
    if show:
        plt.show()

    return dict(
        gamma=gain_gamma, alpha=alpha,
        strip_center_x=strip_xc, strip_width=strip_w, strip_y_span=(strip_y0, strip_y1),
        greedy_time=float(greedy_time), dijkstra_time=float(dijk_time), overlap=float(ov)
    )

if __name__ == "__main__":
    case_hex_with_disks(show=False, seed=42)
    case_square_with_hole(show=False, seed=43)
    case_square_with_vertical_conductive_strip(show=False, seed=44)

    case_hex_with_jam_offsets(show=False)
    plt.show()
