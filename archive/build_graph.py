import geopandas as gpd
"""
build an adjacency list of tuples for each node
"""
roads = gpd.read_file("filtered_roads.geojson")
edges = []
list_nodes = []
V = 0
rep_node = dict()

for ind, road in roads.iterrows():
    u_id, v_id = int(road["NodeIDFrom"]), int(road["NodeIDTo"])
    for x in [u_id, v_id]:
        if x not in rep_node:
            rep_node[x] = V
            list_nodes.append(x)
            V += 1
    u, v = rep_node[u_id], rep_node[v_id]
    dist = float(road["SHAPE_Length"])
    speed = float(road["POSTED_SPEED"]) # for some reason this data can be blank
    time = dist / speed
    dir = road["TrafDir"]
    if dir == "W" or dir == "T":
        edges.append((u, v, time))
    if dir == "A" or dir == "T":
        edges.append((v, u, time))

with open("graph_edges.txt", "w") as file:
    file.write(str(V) + " " + str(len(edges)) + "\n")
    file.write(' '.join(map(str, list_nodes)) + "\n")
    for u, v, w in edges:
        file.write(str(u) + " " + str(v) + " " + str(w) + "\n")
