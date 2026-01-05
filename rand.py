# pick_landmarks_to_nodeids.py
import numpy as np
import geopandas as gpd

NODES = "nodes_points_2263.geojson"          # from Step C
MAN    = "manhattan_boundary_2263.geojson"   # to ensure we stay on Manhattan

# 1) load nodes and Manhattan polygon (EPSG:2263)
nodes = gpd.read_file(NODES).sort_values("index")
man   = gpd.read_file(MAN)
man_union = man.union_all()

# keep only Manhattan nodes (includes coastline nodes)
in_mask = nodes.geometry.apply(man_union.covers)
nodes_manhattan = nodes.loc[in_mask].copy()

# 2) landmarks (lon, lat) in WGS84
times_square = (-73.9855, 40.7580)
columbia_uni = (-73.9626, 40.8075)   # Low Library / College Walk area

# 3) project landmarks to EPSG:2263
pts = gpd.GeoSeries(
    gpd.points_from_xy([times_square[0], columbia_uni[0]],
                       [times_square[1], columbia_uni[1]]),
    crs="EPSG:4326"
).to_crs(nodes_manhattan.crs)

# 4) nearest node helper
coords = np.c_[nodes_manhattan.geometry.x.values, nodes_manhattan.geometry.y.values]
def nearest_node_id(xy):
    d2 = np.sum((coords - np.array(xy))**2, axis=1)
    i = int(np.argmin(d2))
    return int(nodes_manhattan.iloc[i]["NodeID"]), int(nodes_manhattan.iloc[i]["index"])

ts_id, ts_idx = nearest_node_id((pts.iloc[0].x, pts.iloc[0].y))
cu_id, cu_idx = nearest_node_id((pts.iloc[1].x, pts.iloc[1].y))

print("Times Square  → NodeID:", ts_id, " (index:", ts_idx, ")")
print("Columbia Univ → NodeID:", cu_id, " (index:", cu_idx, ")")
