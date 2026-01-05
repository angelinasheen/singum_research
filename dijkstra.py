from heapq import heappush, heappop
import geopandas as gpd

"""
given 2 node ids, receive the nodes to travel to
"""

INF = 1e18
con, id = [], [] #connections for each node, id of node
node = dict() # node of the id
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

"""
takes in 2 node ids
returns a tuple of a list and a float
list is path in order from start to end
float is time it takes
"""

def dijkstra(u_id, v_id): # start, end id
    start, end = node[u_id], node[v_id]
    vis = set([start]) # clear at end
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
        par[u] = INF
    

    return (path, final)


Q = int(input())

for q in range(Q):
    u_id, v_id = map(int, input().split())
    result = dijkstra(u_id, v_id)
    print("path: ", *result[0])
    print("time: ", result[1], "minutes")