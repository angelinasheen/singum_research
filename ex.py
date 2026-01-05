import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.patches import Circle
from shapely.validation import make_valid
from shapely.geometry import LineString, Point
from shapely import union_all

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csgraph
from scipy.interpolate import LinearNDInterpolator, griddata

from case import (
    align_and_clip,
    apply_dirichlet_row_replace,
    build_graph_from_roads,
    dijkstra_path,
    greedy_path_on_nodes,
    load_graph,
    nan_gaussian_blur,
    _interp_K_no_nans,
    _manhattan_mask_and_grid,
    path_time,
    solve_with_component_pinning,
)

# ----------------------- USER KNOBS -----------------------
TARGET_CRS = "EPSG:2263"

# I/O (adjust if your filenames differ)
ROADS_IN  = "filtered_roads.geojson"               # full NYC roads
MAN_IN    = "manhattan_boundary.geojson"           # Manhattan polygon

ROADS_2263 = "filtered_roads_2263.geojson"
MAN_2263   = "manhattan_boundary_2263.geojson"
ROADS_CLIP = "filtered_roads_manhattan_2263.geojson"

# graph + nodes
GRAPH_TXT = "graph_edges.txt"
NODES_GDF = "nodes_points_2263.geojson"

# Voronoi (input cells) & outputs
VORO_IN   = "singum_voronoi.geojson"               # must contain NodeID + geometry (cells)
VORO_BASE = "singum_voronoi_with_K_base.geojson"   # baseline K
VORO_JAM  = "singum_voronoi_with_K_jam.geojson"    # jam-aware K
K_CENTROID_CSV_BASE = "K_centroids_2263_base.csv"
K_CENTROID_CSV_JAM  = "K_centroids_2263_jam.csv"

# Continuum plots
PNG_BASE = "continuum_base.png"
PNG_JAM  = "continuum_jam.png"
PNG_SIDE = "continuum_side_by_side.png"
PNG_JAM_ROUTES = "jam_routes.png"
PNG_JAM_ROUTE_FMT = "jam_routes_case_{case}.png"

# Start/End node IDs (used only to build the graph/edge conductances)
START_NODEID = 21465
END_NODEID   = 78136

# Columbia Neumann ring
COLUMBIA_LONLAT = (-73.9626, 40.8075)
COLUMBIA_RADIUS_FT = 600.0
TOTAL_FLUX_INTO_COLUMBIA = 5000.0
SOURCE_RADIUS_FT = 600.0

# Grid for continuum
GRID_NX, GRID_NY = 900, 1350
GRID_BUFFER_FT = 250.0

# Singum reference length (for paper formula w_I = K^I * A^I / l_p0)
LP0_SINGUM = None   # None => auto-estimate from shared Voronoi face median (fallback road edge median)

# Jam controls (continuum-on-both: rebuild K with jam and solve PDE again)
RUN_JAM      = True
LP0_JAM      = 200.0  # jam radius (feet)
JAM_FACTOR   = 15.0   # slow edges in disk: K^I -> K^I / JAM_FACTOR (i.e., time × JAM_FACTOR)
JAM_OFFSET_Y = 0.0    # move disk center vertically from mid(S,E) by this many *LP0_JAM*
                      # (set to e.g. 2.0 to push it north of the midpoint)
JAM_OFFSET_FACTORS = [0.0, 1/3, 2/3, 1.0, 2.0]  # offsets for figure style sweep (Δy = factor × radius)
RUN_JAM_ROUTE_SWEEP = True                      # compute singum vs. Dijkstra routes for each offset
JAM_ROUTES_GEOJSON = "jam_routes.geojson"
JAM_ROUTE_GRID_NX = GRID_NX // 2
JAM_ROUTE_GRID_NY = GRID_NY // 2

# ----------------------- CONSTANTS -----------------------
MPH_TO_FTPS = 1.46667
SEC_TO_MIN  = 1.0/60.0
CONST_MPH   = 25.0  # all edges 25 mph

# ----------------------- HELPERS -----------------------
def _ord_pair(i, j):
    iu = int(i); jv = int(j)
    return (iu, jv) if iu < jv else (jv, iu)

def _start_xy():
    nodes = gpd.read_file(NODES_GDF).sort_values("index")
    match = nodes[nodes["NodeID"] == START_NODEID]
    if match.empty:
        raise RuntimeError(f"Start NodeID {START_NODEID} not found in {NODES_GDF}")
    geom = match.iloc[0].geometry
    return float(geom.x), float(geom.y)

# ----------------------- K TENSORS (paper rank-1 corrected) -----------------------
def _prep_voronoi():
    if not os.path.exists(VORO_IN):
        raise RuntimeError(f"Missing {VORO_IN}")
    vor_raw = gpd.read_file(VORO_IN)[["NodeID","geometry"]].copy()
    vor_raw["NodeID"] = pd.to_numeric(vor_raw["NodeID"], errors="coerce").astype("Int64")
    man = gpd.read_file(MAN_2263); crs = man.crs
    man_union = make_valid(union_all(list(man.geometry)))
    vor_clip = gpd.overlay(vor_raw, gpd.GeoDataFrame(geometry=[man_union], crs=crs),
                           how="intersection", keep_geom_type=True)
    vor = vor_clip.dissolve(by="NodeID", as_index=False)
    vor["area_ft2"] = vor.geometry.area
    return vor_raw, vor, man_union, crs

def _shared_face_len(poly_by_idx, a_idx, b_idx):
    pa = poly_by_idx.get(a_idx); pb = poly_by_idx.get(b_idx)
    if pa is None or pb is None: return 0.0
    inter = pa.boundary.intersection(pb.boundary)
    try: return float(inter.length)
    except Exception: return 0.0

def _edges_intersecting_disk(coords, efrom, eto, center_xy, radius_ft):
    """Return the set of undirected edge index pairs whose segment intersects a disk."""
    if center_xy is None or radius_ft is None or radius_ft <= 0:
        return set()
    from shapely.prepared import prep
    disk = Point(float(center_xy[0]), float(center_xy[1])).buffer(float(radius_ft))
    pdisk = prep(disk)
    slow = set()
    for u, v in zip(efrom, eto):
        a, b = _ord_pair(u, v)
        if (a, b) in slow:
            continue
        seg = LineString([tuple(coords[u]), tuple(coords[v])])
        if pdisk.intersects(seg):
            slow.add((a, b))
    return slow

def _build_K_generic(voro_out_path, k_centroid_csv_path,
                     jam_center_xy=None, jam_radius_ft=None, jam_factor=1.0):
    """Build K using w_I = (K^I * A^I / l_p0) and K = Σ(w n n^T) - (Σ(w n))(Σ(w n))^T / Σ(w). 
       If jam_* is set, scale conductance inside the disk by 1/jam_factor."""
    vor_raw, vor, man_union, crs = _prep_voronoi()

    # graph + coords
    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)
    nodes = gpd.read_file(NODES_GDF).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    assert V == coords.shape[0] == len(list_nodes)

    nodeid_to_idx = {nid:i for i,nid in enumerate(list_nodes)}
    vor = vor[vor["NodeID"].isin(nodeid_to_idx)].copy()
    vor["gidx"] = vor["NodeID"].map(nodeid_to_idx).astype(int)
    poly_by_idx = {int(r.gidx): r.geometry for _, r in vor.iterrows()}

    # undirected conductance (sum of 1/time), plus straight length
    Gpair = defaultdict(float)
    Lpair = defaultdict(float)
    for u,v,t in zip(efrom, eto, etime):
        if t <= 0: continue
        a,b = _ord_pair(u, v)
        Gpair[(a,b)] += 1.0/t
        if Lpair[(a,b)] == 0.0:
            Lpair[(a,b)] = float(np.linalg.norm(coords[a]-coords[b]))

    # jam mask on pairs via segment–disk intersection
    slow_pair = set()
    if jam_center_xy is not None and jam_radius_ft is not None and jam_factor > 1.0:
        slow_pair = _edges_intersecting_disk(coords, efrom, eto, jam_center_xy, jam_radius_ft)

    # estimate l_p0 if needed
    faces, edges = [], []
    for (a,b) in Gpair.keys():
        fl = _shared_face_len(poly_by_idx, a,b)
        if fl > 0: faces.append(fl)
        else: edges.append(Lpair.get((a,b),0.0))
    if LP0_SINGUM is not None and LP0_SINGUM > 0:
        lp0 = float(LP0_SINGUM)
    else:
        if len(faces)>0: lp0 = float(np.median(faces))
        elif len(edges)>0: lp0 = float(np.median([e for e in edges if e>0] or [1.0]))
        else: lp0 = 1.0
    if lp0 <= 0: lp0 = 1.0

    # neighbor lists
    nbrs = defaultdict(list)
    for (a,b), G in Gpair.items():
        nbrs[a].append((b,G)); nbrs[b].append((a,G))

    Vn = V
    Kxx = np.zeros(Vn); Kxy = np.zeros(Vn); Kyy = np.zeros(Vn)
    area = np.zeros(Vn)
    area[vor["gidx"].to_numpy(int)] = vor["area_ft2"].to_numpy(float)
    eps = 1e-12

    for u in range(Vn):
        Au = area[u]
        if Au <= 0: continue
        pu = coords[u]
        S0 = 0.0; S1 = np.zeros(2); S2 = np.zeros((2,2))
        for v,G in nbrs.get(u, []):
            pv = coords[v]
            d = pv - pu
            L = float(np.linalg.norm(d))
            if L <= eps or G <= 0: continue
            n = d / L
            a,b = _ord_pair(u, v)
            G_eff = G / float(jam_factor) if (a,b) in slow_pair else G
            A_I = _shared_face_len(poly_by_idx, u, v)
            if A_I <= 0: A_I = Lpair.get((a,b), L)
            w = G_eff * (A_I / lp0)
            S0 += w; S1 += w*n; S2 += w*np.outer(n, n)
        if S0 <= eps: continue
        Kmat = S2 - np.outer(S1, S1)/S0
        Kmat /= max(Au, eps)   # per-cell area normalization
        Kxx[u] = float(Kmat[0,0]); Kxy[u] = float(Kmat[0,1]); Kyy[u] = float(Kmat[1,1])

    # write
    out = vor.merge(
        pd.DataFrame({"gidx": np.arange(Vn), "Kxx":Kxx, "Kxy":Kxy, "Kyy":Kyy}),
        on="gidx", how="left"
    )[["NodeID","geometry","Kxx","Kxy","Kyy"]]
    out = out.dissolve(by="NodeID", as_index=False)
    out.to_file(voro_out_path, driver="GeoJSON")

    cent = out.geometry.centroid
    np.savetxt(
        k_centroid_csv_path,
        np.c_[cent.x.values, cent.y.values,
              out["Kxx"].to_numpy(float),
              out["Kxy"].to_numpy(float),
              out["Kyy"].to_numpy(float)],
        delimiter=",", header="x_ft,y_ft,Kxx,Kxy,Kyy", comments=""
    )
    if jam_center_xy is None:
        print(f"[K|base] wrote {voro_out_path}  (l_p0={lp0:.3f})")
    else:
        print(f"[K|jam ] wrote {voro_out_path}  (l_p0={lp0:.3f}, jam_factor={jam_factor})")

def _solve_phi_for_times(V, efrom, eto, etime, src_idx, dst_idx):
    """Solve the network Laplacian with Dirichlet boundary at (src,dst) given edge times."""
    Gpair = defaultdict(float)
    for u, v, t in zip(efrom, eto, etime):
        if t <= 0:
            continue
        a, b = _ord_pair(u, v)
        Gpair[(a, b)] += 1.0 / max(t, 1e-12)

    rows, cols, data = [], [], []
    for (a, b), G in Gpair.items():
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        data += [G, G, -G, -G]
    L = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(V, V))

    ML, b = apply_dirichlet_row_replace(L, [src_idx], 1.0)
    ML, b0 = apply_dirichlet_row_replace(ML, [dst_idx], 0.0)
    b += b0
    return solve_with_component_pinning(ML, b, pinned_nodes=np.array([src_idx, dst_idx], int))

def _phi_on_grid(coords, phi, X, Y, inside, blur_sigma=0.6):
    """Interpolate node potentials to the continuum grid for plotting."""
    Phi = griddata(coords, phi, (X, Y), method="linear")
    nanmask = (~np.isfinite(Phi)) & inside
    if nanmask.any():
        Phi[nanmask] = griddata(coords, phi, (X[nanmask], Y[nanmask]), method="nearest")
    Phi[~inside] = np.nan
    Phi = nan_gaussian_blur(Phi, blur_sigma)
    v0, v1 = np.nanmin(Phi), np.nanmax(Phi)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        return np.zeros_like(Phi)
    return (Phi - v0) / (v1 - v0)

def _plot_single_jam_case(entry, coords, roads, man_union, man_crs, X, Y, inside,
                          src, dst, out_png):
    """Plot gradient background + routes for a single jam case."""
    fig, ax = plt.subplots(figsize=(8.0, 10.5))
    Vn = entry["phi_grid"]
    im = ax.pcolormesh(X, Y, Vn, shading="auto", cmap="viridis", vmin=0, vmax=1, zorder=1)
    ax.contour(X, Y, Vn, levels=16, colors='k', linewidths=0.45, alpha=0.45, zorder=2)
    gpd.GeoDataFrame(geometry=[man_union], crs=man_crs).boundary.plot(
        ax=ax, color="black", linewidth=0.9, zorder=4
    )
    if roads is not None:
        roads.plot(ax=ax, color="#f5f5f5", linewidth=0.35, zorder=3)

    circ = Circle(
        (entry["center"][0], entry["center"][1]),
        entry["radius"],
        facecolor="gold",
        edgecolor="orange",
        lw=1.0,
        alpha=0.28,
        zorder=5,
    )
    ax.add_patch(circ)

    if entry["sing_nodes"]:
        C = coords[np.array(entry["sing_nodes"], int)]
        ax.plot(C[:, 0], C[:, 1], "-", lw=3.0, color="#00accf", label="Singum greedy", zorder=6)
    if entry["dijk_nodes"]:
        C = coords[np.array(entry["dijk_nodes"], int)]
        ax.plot(C[:, 0], C[:, 1], "-", lw=3.0, color="#d81b60", label="Dijkstra", zorder=7)
    ax.plot(coords[src, 0], coords[src, 1], "go", ms=8, zorder=8)
    ax.plot(coords[dst, 0], coords[dst, 1], "ro", ms=8, zorder=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (ft, EPSG:2263)")
    ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title(f"Jam case {entry['case']}: Δy={entry['offset_factor']:.2f}×R")
    ax.legend(loc="upper right")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Node potential φ")
    cb.ax.tick_params(labelsize=8)
    ax.text(
        0.02,
        0.02,
        f"Singum {entry['sing_time']:.1f} min\nDijkstra {entry['dijk_time']:.1f} min",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Saved:", out_png)

def _compute_jam_routes_and_plot(
    offsets=JAM_OFFSET_FACTORS,
    jam_radius_ft=LP0_JAM,
    jam_factor=JAM_FACTOR,
    out_geojson=JAM_ROUTES_GEOJSON,
    out_png=PNG_JAM_ROUTES,
):
    """Compute Singum vs. Dijkstra routes for several jam offsets and save GeoJSON + figure."""
    if not offsets or jam_radius_ft is None or jam_radius_ft <= 0:
        return
    nodes = gpd.read_file(NODES_GDF).sort_values("index")
    coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]

    V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)
    assert coords.shape[0] == V, "Node count mismatch when building jam routes."
    idx = {nid: i for i, nid in enumerate(list_nodes)}
    src = idx[START_NODEID]; dst = idx[END_NODEID]
    mid_xy = 0.5 * (coords[src] + coords[dst])

    case_entries = []
    rows, geoms = [], []
    man = gpd.read_file(MAN_2263)
    man_union = make_valid(union_all(list(man.geometry)))
    man_crs = man.crs
    _, Xg, Yg, inside = _manhattan_mask_and_grid(
        nx=max(50, JAM_ROUTE_GRID_NX), ny=max(50, JAM_ROUTE_GRID_NY), extra_buffer_ft=GRID_BUFFER_FT
    )

    for case_id, offset in enumerate(offsets, start=1):
        center = np.array([mid_xy[0], mid_xy[1] + float(offset) * jam_radius_ft], float)
        slow_pairs = _edges_intersecting_disk(coords, efrom, eto, center, jam_radius_ft) if jam_factor>1.0 else set()
        etime_case = etime.copy()
        if jam_factor > 1.0 and slow_pairs:
            for i, (u, v) in enumerate(zip(efrom, eto)):
                if _ord_pair(u, v) in slow_pairs:
                    etime_case[i] *= jam_factor

        phi = _solve_phi_for_times(V, efrom, eto, etime_case, src, dst)
        phi_grid = _phi_on_grid(coords, phi, Xg, Yg, inside)
        tmap = {(u, v): w for u, v, w in zip(efrom, eto, etime_case)}
        sing_nodes = greedy_path_on_nodes(phi, efrom, eto, src, dst, coords=coords)
        dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime_case, src, dst)
        sing_time = path_time(sing_nodes, tmap)

        def _geom_from_nodes(path):
            if len(path) < 2:
                return None
            return LineString([tuple(coords[i]) for i in path])

        for method, path_nodes, trip_time in (
            ("singum", sing_nodes, sing_time),
            ("dijkstra", dijk_nodes, dijk_time),
        ):
            geom = _geom_from_nodes(path_nodes)
            rows.append(
                {"method": method, "case": case_id, "offset_factor": float(offset), "time_min": float(trip_time)}
            )
            geoms.append(geom)

        case_entries.append(
            {
                "case": case_id,
                "offset_factor": float(offset),
                "center": center,
                "radius": jam_radius_ft,
                "sing_nodes": sing_nodes,
                "dijk_nodes": dijk_nodes,
                "sing_time": float(sing_time),
                "dijk_time": float(dijk_time),
                "phi_grid": phi_grid,
            }
        )
        print(
            f"[Jam case {case_id}] offset={offset:.3f}×R  singum={sing_time:.2f} min  "
            f"dijkstra={dijk_time:.2f} min  slowed_edges={len(slow_pairs)}"
        )

    jam_gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=man_crs).dropna(subset=["geometry"])
    if not jam_gdf.empty:
        jam_gdf.to_file(out_geojson, driver="GeoJSON")
        print("Saved:", out_geojson)
    else:
        print("Jam routes GeoDataFrame empty; nothing written.")

    # Plot each case in a row for Figure 4 style comparison
    try:
        roads = None
        if os.path.exists(ROADS_CLIP):
            roads = gpd.read_file(ROADS_CLIP)
        ncols = len(case_entries)
        fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 7.6), sharey=True)
        if ncols == 1:
            axes = [axes]
        im_last = None
        for ax, entry in zip(axes, case_entries):
            Vn = entry["phi_grid"]
            im_last = ax.pcolormesh(Xg, Yg, Vn, shading="auto", cmap="viridis", vmin=0, vmax=1, zorder=1)
            ax.contour(Xg, Yg, Vn, levels=16, colors='k', linewidths=0.4, alpha=0.45, zorder=2)
            if roads is not None:
                roads.plot(ax=ax, color="#f5f5f5", linewidth=0.3, zorder=3)
            gpd.GeoDataFrame(geometry=[man_union], crs=man_crs).boundary.plot(
                ax=ax, color="black", linewidth=0.7, zorder=4
            )
            circ = Circle(
                (entry["center"][0], entry["center"][1]),
                entry["radius"],
                facecolor="gold",
                edgecolor="orange",
                lw=1.0,
                alpha=0.28,
                zorder=5,
            )
            ax.add_patch(circ)
            if entry["sing_nodes"]:
                C = coords[np.array(entry["sing_nodes"], int)]
                ax.plot(C[:, 0], C[:, 1], "-", lw=2.7, color="#00accf", label="Singum greedy", zorder=6)
            if entry["dijk_nodes"]:
                C = coords[np.array(entry["dijk_nodes"], int)]
                ax.plot(C[:, 0], C[:, 1], "-", lw=2.7, color="#d81b60", label="Dijkstra", zorder=7)
            ax.plot(coords[src, 0], coords[src, 1], "go", ms=6, zorder=8)
            ax.plot(coords[dst, 0], coords[dst, 1], "ro", ms=6, zorder=8)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Case {entry['case']}: Δy={entry['offset_factor']:.2f}×R")
            ax.text(
                0.02,
                0.02,
                f"Singum {entry['sing_time']:.1f} min\nDijkstra {entry['dijk_time']:.1f} min",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65),
            )
            if entry["case"] == 1:
                ax.legend(loc="upper right", fontsize=8)
        axes_list = axes if isinstance(axes, list) else list(np.ravel(axes))
        if im_last is not None:
            cbar = fig.colorbar(im_last, ax=axes_list, fraction=0.025, pad=0.02)
            cbar.set_label("Node potential φ")
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print("Saved:", out_png)
    except Exception as exc:
        print("Jam route plotting failed:", exc)

    # Save per-case PNGs with gradient background
    for entry in case_entries:
        single_png = PNG_JAM_ROUTE_FMT.format(case=entry["case"])
        _plot_single_jam_case(entry, coords, roads, man_union, man_crs, Xg, Yg, inside,
                              src, dst, single_png)

# ----------------------- Continuum solver (uses a given K file) -----------------------
def _compute_continuum_field(vor, source_center_xy, source_radius_ft):
    man_union, X, Y, inside = _manhattan_mask_and_grid(
        nx=GRID_NX, ny=GRID_NY, extra_buffer_ft=GRID_BUFFER_FT
    )
    Kxxg, Kxyg, Kyyg = _interp_K_no_nans(vor, X, Y, inside, blur_sigma=0.8)

    # shoreline mask (used only for identifying boundary cells)
    sh = np.zeros_like(inside, dtype=bool)
    sh |= inside & np.roll(~inside,  1, axis=0)
    sh |= inside & np.roll(~inside, -1, axis=0)
    sh |= inside & np.roll(~inside,  1, axis=1)
    sh |= inside & np.roll(~inside, -1, axis=1)
    shoreline = sh

    nxg = X.shape[1]; nyg = X.shape[0]
    dx = (X[0,-1]-X[0,0])/(nxg-1); dy = (Y[-1,0]-Y[0,0])/(nyg-1)
    cell_area = dx * dy
    N  = nxg*nyg

    # Columbia Neumann ring (sink)
    man_crs = gpd.read_file(MAN_2263).crs
    def lonlat_to_2263(lonlat):
        return gpd.GeoSeries([Point(*lonlat)], crs="EPSG:4326").to_crs(man_crs).iloc[0]
    col_pt = lonlat_to_2263(COLUMBIA_LONLAT)
    col_circle = col_pt.buffer(COLUMBIA_RADIUS_FT)
    L_col = float(col_circle.exterior.length)
    from shapely.prepared import prep
    ring = prep(col_circle.boundary.buffer(0.6*max(dx,dy)))
    C_mask = np.zeros_like(inside, dtype=bool)
    xs = X[0,:]; ys = Y[:,0]
    for j in range(0, nyg, max(1, nyg//200)):
        C_mask[j, np.array([ring.contains(Point(x, ys[j])) for x in xs])] = True
    for i in range(0, nxg, max(1, nxg//200)):
        col = np.array([ring.contains(Point(xs[i], y)) for y in ys])
        C_mask[col, i] = True
    C_mask &= inside

    # Source disk mask
    src_mask = np.zeros_like(inside, dtype=bool)
    src_density = 0.0
    if source_center_xy is not None and source_radius_ft and source_radius_ft > 0:
        cx, cy = source_center_xy
        r2 = float(source_radius_ft)**2
        src_mask = inside & (((X - cx)**2 + (Y - cy)**2) <= r2)
        src_cells = int(src_mask.sum())
        if src_cells > 0:
            src_density = float(TOTAL_FLUX_INTO_COLUMBIA) / max(src_cells * cell_area, 1e-12)

    A_rows, A_cols, A_data = [], [], []
    bvec = np.zeros(N, float)
    qn   = -float(TOTAL_FLUX_INTO_COLUMBIA)/max(L_col,1e-12)
    def I(i,j): return j*nxg + i

    ref_idx = None
    for j in range(nyg):
        for i in range(nxg):
            p = I(i,j)
            if not inside[j,i]:
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=0.0
                continue
            if ref_idx is None:
                ref_idx = p
                A_rows.append(p); A_cols.append(p); A_data.append(1.0); bvec[p]=0.0
                continue

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
                if nKn <= 0: continue
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
            if src_density > 0.0 and src_mask[j,i]:
                bvec[p] += src_density * cell_area

    A = csr_matrix((np.array(A_data), (np.array(A_rows), np.array(A_cols))), shape=(N,N))
    eps = 1e-9
    A = A + csr_matrix((np.full(N, eps), (np.arange(N), np.arange(N))), shape=(N,N))

    if not np.isfinite(A.data).all(): A.data[~np.isfinite(A.data)] = 0.0
    if not np.isfinite(bvec).all():   bvec[~np.isfinite(bvec)] = 0.0

    V = spsolve(A, bvec).reshape(Y.shape)
    return man_union, X, Y, inside, V

def solve_continuum_and_plot_using(
    voro_path, out_png, title, vmin=0.0, vmax=1.0, source_center_xy=None, source_radius_ft=SOURCE_RADIUS_FT
):
    vor = gpd.read_file(voro_path)
    if source_center_xy is None:
        source_center_xy = _start_xy()
    man_union, X, Y, inside, V = _compute_continuum_field(vor, source_center_xy, source_radius_ft)

    # plot with FIXED normalization [vmin,vmax]
    V_plot = V.copy(); V_plot[~inside] = np.nan
    if np.isfinite(vmin) and np.isfinite(vmax) and (vmax > vmin):
        Vn = (V_plot - vmin) / max(vmax - vmin, 1e-12)
    else:
        v0, v1 = np.nanmin(V_plot), np.nanmax(V_plot)
        Vn = (V_plot - v0) / max(v1 - v0, 1e-12)

    fig, ax = plt.subplots(figsize=(8.8, 11.2))
    gpd.GeoDataFrame(geometry=[man_union], crs=gpd.read_file(MAN_2263).crs)\
      .boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=1)

    im = ax.pcolormesh(X, Y, Vn, shading="auto", cmap="viridis", zorder=2, vmin=0, vmax=1)
    ax.contour(X, Y, Vn, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)
    try:
        gpd.read_file(ROADS_CLIP).plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
    except Exception:
        pass

    ax.set_aspect('equal'); ax.set_xlabel("x (ft, EPSG:2263)"); ax.set_ylabel("y (ft, EPSG:2263)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Continuum potential φ (fixed scale)")
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close(fig)
    print("Saved:", out_png)

# ----------------------- MAIN FLOW -----------------------
def main():
    # 1) Align + clip once
    if not os.path.exists(ROADS_CLIP):
        align_and_clip()

    # 2) Build graph + nodes (for conductances & geometry)
    if not (os.path.exists(GRAPH_TXT) and os.path.exists(NODES_GDF)):
        build_graph_from_roads()
    
    #print("exit here at a")
    #exit()

    if RUN_JAM_ROUTE_SWEEP:
        _compute_jam_routes_and_plot()

    print("exit here at b")
    exit()

    # 3) Baseline K (paper formula, no jam) → continuum
    _build_K_generic(VORO_BASE, K_CENTROID_CSV_BASE, jam_center_xy=None, jam_radius_ft=None, jam_factor=1.0)
    solve_continuum_and_plot_using(
        voro_path=VORO_BASE,
        out_png=PNG_BASE,
        title="Continuum potential on Manhattan (baseline K)",
        vmin=0.0,
        vmax=1.0,
        source_center_xy=_start_xy(),
        source_radius_ft=SOURCE_RADIUS_FT,
    )
    print("RUN_JAM value: ", RUN_JAM)

    print("exit here at c")
    exit()
    if RUN_JAM:
        # 4) Jam center near midpoint of (src,dst)
        V, list_nodes, efrom, eto, etime = load_graph(GRAPH_TXT)
        nodes = gpd.read_file(NODES_GDF).sort_values("index")
        coords = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
        idx = {nid:i for i,nid in enumerate(list_nodes)}
        src = idx[START_NODEID]; dst = idx[END_NODEID]
        mid_xy = 0.5*(coords[src] + coords[dst])

        lp0_jam = float(LP0_JAM if (LP0_JAM and LP0_JAM>0) else np.median(np.linalg.norm(coords[efrom]-coords[eto], axis=1)))
        center = np.array([mid_xy[0], mid_xy[1] + JAM_OFFSET_Y*lp0_jam], float)
        print("lp0 jam : ",lp0_jam)
        print("center: ", center)

        # 5) Jam-aware K → continuum
        _build_K_generic(VORO_JAM, K_CENTROID_CSV_JAM,
                         jam_center_xy=center, jam_radius_ft=lp0_jam, jam_factor=JAM_FACTOR)
        
        solve_continuum_and_plot_using(
            voro_path=VORO_JAM,
            out_png=PNG_JAM,
            title=f"Continuum potential (jam: radius={lp0_jam:.0f} ft, factor×{JAM_FACTOR})",
            vmin=0.0,
            vmax=1.0,
            source_center_xy=center,
            source_radius_ft=lp0_jam,
        )

        # 6) Optional side-by-side for quick visual compare (both continuum, fixed scale)
        try:
            roads = None
            if os.path.exists(ROADS_CLIP):
                roads = gpd.read_file(ROADS_CLIP)
            specs = [
                (VORO_JAM, center, lp0_jam, f"Jam (r={lp0_jam:.0f} ft, ×{JAM_FACTOR})"),
                (VORO_BASE, _start_xy(), SOURCE_RADIUS_FT, "Baseline"),
            ]
            fig, axes = plt.subplots(1, 2, figsize=(16.5, 10))
            for ax, (vp, src_ctr, src_rad, ttl) in zip(axes, specs):
                vor = gpd.read_file(vp)
                man_union, X, Y, inside, V = _compute_continuum_field(vor, src_ctr, src_rad)
                V_plot = V.copy(); V_plot[~inside] = np.nan
                Vn = (V_plot - np.nanmin(V_plot)) / max(np.nanmax(V_plot) - np.nanmin(V_plot), 1e-12)
                gpd.GeoDataFrame(geometry=[man_union], crs=gpd.read_file(MAN_2263).crs).boundary.plot(
                    ax=ax, color="black", linewidth=1.0, zorder=1
                )
                im = ax.pcolormesh(X, Y, Vn, shading="auto", cmap="viridis", zorder=2, vmin=0, vmax=1)
                ax.contour(X, Y, Vn, levels=16, colors='k', linewidths=0.55, alpha=0.55, zorder=3)
                if roads is not None:
                    roads.plot(ax=ax, color="#eaeaea", linewidth=0.35, zorder=4)
                ax.set_aspect('equal'); ax.set_title(ttl)
                ax.set_xlabel("x (ft, EPSG:2263)"); ax.set_ylabel("y (ft, EPSG:2263)")
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
            cbar.set_label("Continuum potential φ (scaled)")
            fig.tight_layout(); fig.savefig(PNG_SIDE, dpi=220); plt.close(fig)
            print("Saved:", PNG_SIDE)
        except Exception as e:
            print("Side-by-side failed:", e)

if __name__ == "__main__":
    main()
