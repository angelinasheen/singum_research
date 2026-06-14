import geopandas as gpd


def load_lion_gdf():
    path = "data/lion/lion.gdb"
    gdf = gpd.read_file(path, layer="lion")
    columns = ["geometry", "POSTED_SPEED", "TrafDir", "NodeIDFrom", "NodeIDTo", "SegmentID"]
    return gdf[columns]

def nyc_boundaries():
    path = "data/nyc_boundary.geojson"
    gdf = gpd.read_file(path)
    manhattan_gdf = gdf[gdf["district"].str.contains("MN", case=False)]
    manhattan_gdf.to_file("data/manhattan_boundary.geojson", driver="GeoJSON")

