import geopandas as gpd
import pandas as pd 
from load_data import load_lion_gdf, nyc_boundaries
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from collections import defaultdict
import math
import itertools
from rasterstats import zonal_stats
from fiona import listlayers
import numpy as np

#read data
gdf = load_lion_gdf()
# ----------------------------------- DRAWING SINGUMS --------------------------------------

#find coordinates of each node + whether it's start/end node 
def extract_coord(geom, from_end=True):
    if isinstance(geom, LineString):
        coords = list(geom.coords)
    elif isinstance(geom, MultiLineString):
        coords = list(geom.geoms[0].coords) if from_end else list(geom.geoms[-1].coords)
    else:
        return None
    return coords[0] if from_end else coords[-1]

#for each row segment, get beg. point of segment
gdf["From_Coord"] = gdf.apply(lambda row: extract_coord(row.geometry, True), axis=1)
#for each row segment, get end point of segment 
gdf["To_Coord"] = gdf.apply(lambda row: extract_coord(row.geometry, False), axis=1)

#map nodes to segments and coordinates
node_to_segments = defaultdict(list)
node_to_coord = {}

#when this for loop ends every node should be mapped to its start nodes and end nodes
for _, row in gdf.iterrows(): #each row in gdf is a road segment
    node_to_segments[row.NodeIDFrom].append((row.geometry, row.To_Coord)) #for each start node of segment add end and geo
    node_to_segments[row.NodeIDTo].append((row.geometry, row.From_Coord)) #for each end point add start point and geo
    node_to_coord[row.NodeIDFrom] = row.From_Coord #assign coordinate to node
    node_to_coord[row.NodeIDTo] = row.To_Coord 


#build array of coordinates and match index to NodeID
node_coords = np.array([coord for coord in node_to_coord.values()]) #need this to use voronoi package
node_ids = list(node_to_coord.keys()) #node_coords[i] now coresp. to node_ids[i]
id_to_index = {node_ids[i]: i for i in range(len(node_ids))} #each node id is now mapped to its respective index in node coords

#build Voronoi polygons, ignoring infinite regions 
unique_coords, unique_indices = np.unique(node_coords, axis=0, return_index=True)
unique_node_ids = [node_ids[i] for i in unique_indices]
vor = Voronoi(unique_coords)

poly_geoms = []
poly_ids = []
skipped = 0

for point_idx, region_idx in enumerate(vor.point_region):
    verts = vor.regions[region_idx]
    if -1 in verts or len(verts) == 0:
        skipped += 1
        continue
    try:
        polygon = Polygon([vor.vertices[i] for i in verts])
        if polygon.is_valid and not polygon.is_empty:
            poly_geoms.append(polygon)
            poly_ids.append(unique_node_ids[point_idx])
    except:
        skipped += 1

assert len(poly_geoms) + skipped == len(unique_node_ids)


#create GeoDataFrame
voronoi_gdf = gpd.GeoDataFrame({
    "NodeID": poly_ids,
    "geometry": poly_geoms
}, crs=gdf.crs)  # make sure CRS matches original
voronoi_gdf = voronoi_gdf.dissolve(by="NodeID", as_index=False)
manhattan_gdf = gpd.read_file("manhattan_boundary.geojson")

filtered_voronoi = voronoi_gdf[voronoi_gdf.geometry.intersects(manhattan_gdf.iloc[0].geometry)]
filtered_voronoi.to_file("singum_voronoi.geojson", driver="GeoJSON")

print("Before clipping:", len(poly_geoms))
print("After clipping:", len(filtered_voronoi))
print("Unique NodeIDs after clipping:", filtered_voronoi['NodeID'].nunique())
