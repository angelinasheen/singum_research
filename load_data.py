import geopandas as gpd
import rasterio
from rasterio.mask import mask
import rasterio
from rasterio.enums import Resampling


def load_lion_gdf():
    path = "lion/lion.gdb"
    gdf = gpd.read_file(path, layer="lion")
    columns = ["geometry", "POSTED_SPEED", "TrafDir", "NodeIDFrom", "NodeIDTo", "SegmentID"]
    return gdf[columns]

def nyc_boundaries():
    path = "nyc_boundary.geojson"
    gdf = gpd.read_file(path)
    manhattan_gdf = gdf[gdf["district"].str.contains("MN", case=False)]
    manhattan_gdf.to_file("manhattan_boundary.geojson", driver="GeoJSON")

