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

# Voronoi
VORO_IN   = "singum_voronoi.geojson"               # must contain NodeID + geometry
VORO_OUT  = "singum_voronoi_with_K.geojson"
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

# Grid for continuum
GRID_NX, GRID_NY = 700, 1050
GRID_BUFFER_FT = 250.0

# ----------------------- CONSTANTS -----------------------
MPH_TO_FTPS = 1.46667
SEC_TO_MIN  = 1.0/60.0
CONST_MPH   = 25.0  # all edges 25 mph

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

def path_time(path, time_map):
    if len(path) < 2: return np.nan
    t=0.0
    for a,b in zip(path[:-1], path[1:]):
        w = time_map.get((a,b)) or time_map.get((b,a))
        if w is None: return np.nan
        t += w
    return t

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

if __name__ == "__main__":
    main()