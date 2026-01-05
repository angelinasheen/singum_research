import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

# ---------- lattice builder (hex/tri via staggered rows) ----------
def build_tri_lattice(L=2.0, W=2.0, lp0=0.01):
    """
    Returns:
      coords : (N,2) node coordinates
      edges  : (E,2) undirected edges as node indices
    Spacing: dx = 2*lp0 (horiz), dy = sqrt(3)*lp0 (vert)
    """
    dx = 2*lp0
    dy = np.sqrt(3.0)*lp0
    nx = int(round(L/dx) + 1)
    ny = int(round(W/dy) + 1)

    coords = np.zeros((nx*ny, 2), float)
    node_map = np.zeros((ny, nx), dtype=int)

    nid = 0
    for j in range(ny):
        for i in range(nx):
            x = i*dx + (lp0 if (j % 2) == 1 else 0.0)   # stagger every other row
            y = j*dy
            coords[nid] = (x, y)
            node_map[j, i] = nid
            nid += 1

    edges = []
    for j in range(ny):
        for i in range(nx):
            n = node_map[j, i]
            # horizontal neighbor
            if i < nx - 1:
                edges.append((n, node_map[j, i+1]))
            # down neighbors
            if j < ny - 1:
                edges.append((n, node_map[j+1, i]))
                if (j % 2) == 1 and i < nx - 1:
                    edges.append((n, node_map[j+1, i+1]))
                elif (j % 2) == 0 and i > 0:
                    edges.append((n, node_map[j+1, i-1]))

    return coords, np.asarray(edges, dtype=int)

# ---------- params ----------
L = W = 2.0
lp0 = 0.02            # lattice density (smaller -> denser)
R   = 0.05            # disk radius
src_c = (0.30, W/2)   # source disk center
dst_c = (L-0.30, W/2) # sink disk center

# build lattice
coords, edges = build_tri_lattice(L=L, W=W, lp0=lp0)

# nodes inside source/sink disks (for highlighting only)
r_src = np.linalg.norm(coords - np.array(src_c)[None, :], axis=1)
r_dst = np.linalg.norm(coords - np.array(dst_c)[None, :], axis=1)
src_nodes = np.where(r_src <= R)[0]
dst_nodes = np.where(r_dst <= R)[0]

# segments for drawing
segs = np.stack([coords[edges[:,0]], coords[edges[:,1]]], axis=1)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7.2, 6))
lc = LineCollection(segs, colors=(0.75, 0.75, 0.75, 1.0), linewidths=0.5)
ax.add_collection(lc)
ax.autoscale()

# lattice nodes (light)
ax.plot(coords[:,0], coords[:,1], '.', ms=2, color='0.55', alpha=0.8, label='nodes')

# highlight source/sink nodes
if src_nodes.size:
    ax.plot(coords[src_nodes,0], coords[src_nodes,1], 'r.', ms=4, label='source nodes')
if dst_nodes.size:
    ax.plot(coords[dst_nodes,0], coords[dst_nodes,1], 'c.', ms=4, label='sink nodes')

# draw the disks and domain box
ax.add_patch(Circle(src_c, R, fill=False, ec='r', lw=2, label='source disk'))
ax.add_patch(Circle(dst_c, R, fill=False, ec='c', lw=2, label='sink disk'))
ax.add_patch(Rectangle((0,0), L, W, fill=False, ec='k', lw=1.2))

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.02, L+0.02); ax.set_ylim(-0.02, W+0.02)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
