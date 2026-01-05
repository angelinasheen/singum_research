from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import matplotlib.pyplot as plt

# i literally stole this from
# https://github.com/codeforgermany/click_that_hood/blob/main/public/data/manhattan.geojson

gdf = gpd.read_file("manhattan.geojson")

manhattan = gdf.iloc[0]["geometry"]
manhattan_gdf = gpd.GeoDataFrame(geometry=[manhattan], crs="EPSG:4326")

fig, ax = plt.subplots()

gdf.plot(ax=ax, facecolor=None, edgecolor=None, linewidth=1)
print(gdf.columns)

# create a geojson out of the labeled singums
# we need just an outline of manhattan

plt.title("testing")
plt.show()