import geopandas as gpd
import fiona

gdb_path = "lion/lion.gdb"
layers = fiona.listlayers(gdb_path)

print(layers)
# nodes = gpd.read_file(gdb_path, layer="node")
# the EPSG is 2263, this requires conversion
# nodes.to_file("nodes.geojson", driver="GeoJSON")

roads = gpd.read_file(gdb_path, layer="lion")
print("read")
# miles
roads = roads.to_crs(epsg=4326)
roads["SHAPE_Length"] = roads["SHAPE_Length"].div(5280)

roads["POSTED_SPEED"] = roads["POSTED_SPEED"].str.strip() # this could be empty
roads = roads[roads["POSTED_SPEED"].str.isdigit()]
roads["POSTED_SPEED"] = roads["POSTED_SPEED"].astype(int).div(60)

print("data_fix")

filtered_roads = roads[[
    "TrafDir",
    "NodeIDFrom",
    "NodeIDTo",
    "POSTED_SPEED",
    "geometry",
    "SHAPE_Length"
]]

print("data_filter")

filtered_roads.to_file("filtered_roads.geojson", driver="GeoJSON")