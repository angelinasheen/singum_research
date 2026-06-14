from heapq import heappush, heappop
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

INF = 1e18
con, id = [], []
node = dict()
V, E = 0, 0

with open("graph_edges.txt", "r") as file:
    V, E = map(int, file.readline().split())
    con = [[] for _ in range(V)]
    id = list(map(int, file.readline().split()))
    for i, x in enumerate(id):
        node[x] = i

    for e in range(E):
        u, v, d = file.readline().split()
        u, v, d = int(u), int(v), float(d)
        con[u].append((v, d))
        con[v].append((u, d))

par = [-1] * V
dist = [INF] * V

def dijkstra(u_id, v_id):
    start, end = node[u_id], node[v_id]
    vis = set([start])
    heap = [(0, start)]
    global par, dist
    dist[start] = 0
    while heap and par[end] == -1:
        (d, u) = heappop(heap)
        if d != dist[u]: continue
        for v, nd in con[u]:
            if d + nd < dist[v]:
                dist[v] = d + nd
                heappush(heap, (dist[v], v))
                par[v] = u
                vis.add(v)
    if par[end] == -1:
        return ([], INF)
    path = []
    final = dist[end]
    while end != -1:
        path.append(id[end])
        end = par[end]
    path.reverse()
    for u in vis:
        dist[u] = INF
        par[u] = -1
    return (path, final)

roads = gpd.read_file("filtered_roads.geojson")
gdf = gpd.read_file("manhattan.geojson")

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Dijkstra Visualization')
plt.subplots_adjust(left=0.4)
gdf.plot(ax=ax, facecolor=None, edgecolor=None, linewidth=1)
roads.plot(ax=ax, color="gray", linewidth=0.5)

ax_start = plt.axes([0.15, 0.7, 0.1, 0.05])
start_box = TextBox(ax_start, 'Start ID: ')
ax_end = plt.axes([0.15, 0.5, 0.1, 0.05])
end_box = TextBox(ax_end, 'End ID: ')

ax_button = plt.axes([0.15, 0.3, 0.1, 0.05])
button = Button(ax_button, 'Run')

def run_dijkstra(event):
    try:
        start_id = int(start_box.text)
        end_id = int(end_box.text)
        result = dijkstra(start_id, end_id)
        path, time = result
        idFrom = [int(x) for x in roads["NodeIDFrom"].tolist()]
        idTo = [int(x) for x in roads["NodeIDTo"].tolist()]
        ax.clear()
        gdf.plot(ax=ax, facecolor=None, edgecolor=None, linewidth=1)
        roads.plot(ax=ax, color="gray", linewidth=0.5)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for j in range(len(roads)):
                if (idFrom[j] == u and idTo[j] == v) or (idFrom[j] == v and idTo[j] == u):
                    roads.iloc[[j]].plot(ax=ax, color="red", linewidth=2)
                    break
        print("Path: ", *path)
        print(str(time) + " minutes")
    except ValueError:
        print("Invalid input. Please enter valid node IDs.")
button.on_clicked(run_dijkstra)

plt.show()