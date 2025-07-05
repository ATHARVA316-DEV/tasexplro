import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq
import random

# =============== CONFIGURATION ===============
NX, NY, NZ = 30, 30, 15   # grid dimensions (x, y, z)
NUM_CITIES = 8            # number of cities
NUM_OBS    = 4            # number of cuboid obstacles
OBS_MAX_W  = 8            # obstacle max width  (x)
OBS_MAX_H  = 8            # obstacle max height (y)
OBS_MAX_D  = 6            # obstacle max depth  (z)
SEED       = 4            # random seed for repeatability (None for fresh)
# ============================================

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

# ------------------------------------------------------------
# Helper: Generate a random cuboid obstacle (axis‑aligned)
def place_cuboid(grid, x0, y0, z0, w, h, d):
    grid[z0:z0+d, y0:y0+h, x0:x0+w] = 1
    return (x0, y0, z0, w, h, d)

# -------------- Build grid with obstacles -------------------
grid = np.zeros((NZ, NY, NX), dtype=np.int8)  # 0=free, 1=obstacle
obstacles = []
for _ in range(NUM_OBS):
    w = random.randint(4, OBS_MAX_W)
    h = random.randint(4, OBS_MAX_H)
    d = random.randint(3, OBS_MAX_D)
    x0 = random.randint(0, NX - w - 1)
    y0 = random.randint(0, NY - h - 1)
    z0 = random.randint(0, NZ - d - 1)
    obstacles.append(place_cuboid(grid, x0, y0, z0, w, h, d))

# -------------- Generate cities (free voxels) ---------------
cities = []
while len(cities) < NUM_CITIES:
    x, y, z = random.randint(0, NX-1), random.randint(0, NY-1), random.randint(0, NZ-1)
    if grid[z, y, x] == 0 and (x,y,z) not in cities:
        cities.append((x, y, z))
cities = np.array(cities)

# -------------- 3‑D A* (6‑neighbour) ------------------------
dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
def astar(start, goal):
    sx, sy, sz = start
    gx, gy, gz = goal
    open_heap = []
    heapq.heappush(open_heap, (abs(sx-gx)+abs(sy-gy)+abs(sz-gz), 0, start))
    came_from = {}
    gscore = {start: 0}
    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        cx, cy, cz = current
        for dx, dy, dz in dirs:
            nx, ny, nz = cx+dx, cy+dy, cz+dz
            if 0 <= nx < NX and 0 <= ny < NY and 0 <= nz < NZ and grid[nz, ny, nx] == 0:
                ng = g + 1
                nxt = (nx, ny, nz)
                if ng < gscore.get(nxt, 1e9):
                    gscore[nxt] = ng
                    h = abs(nx-gx)+abs(ny-gy)+abs(nz-gz)
                    heapq.heappush(open_heap, (ng+h, ng, nxt))
                    came_from[nxt] = current
    return None  # should not happen in sparse worlds

# -------------- Distance & path matrices --------------------
N = NUM_CITIES
dist = np.zeros((N, N))
paths = [[None]*N for _ in range(N)]
for i in range(N):
    for j in range(i+1, N):
        p = astar(tuple(cities[i]), tuple(cities[j]))
        if p is None:
            raise RuntimeError("Unreachable city pair – obstacle layout too dense.")
        d = len(p) - 1
        dist[i, j] = dist[j, i] = d
        paths[i][j] = paths[j][i] = p

# -------------- TSP utilities --------------------------------
def tour_length(route):
    return sum(dist[route[k], route[(k+1)%N]] for k in range(N))

def two_opt(route):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, N-2):
            for j in range(i+1, N):
                if j - i == 1:
                    continue
                new = best[:]
                new[i:j] = best[i:j][::-1]
                if tour_length(new) < tour_length(best):
                    best = new
                    improved = True
        route = best
    return best

# -------------- Initial & optimized tours --------------------
init_route = list(range(N))
random.shuffle(init_route)
opt_route  = two_opt(init_route)
init_len   = tour_length(init_route)
opt_len    = tour_length(opt_route)

# -------------- 3‑D visualisation ----------------------------
def plot_world(ax, route, title):
    # plot obstacles as translucent voxels
    ax.voxels(grid, facecolors='lightgrey', edgecolor='k', alpha=0.25)
    # plot cities
    xs, ys, zs = cities[:,0], cities[:,1], cities[:,2]
    ax.scatter(xs, ys, zs, s=60, c='yellow', edgecolor='black', depthshade=True)
    for idx, (x,y,z) in enumerate(cities):
        ax.text(x+0.5, y+0.5, z+0.5, str(idx), color='blue', fontsize=8)
    # plot route segments
    for k in range(N):
        a = route[k]
        b = route[(k+1)%N]
        path = paths[a][b]
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        pz = [p[2] for p in path]
        ax.plot(px, py, pz, linewidth=1.5)
    ax.set_title(f"{title}\nLength = {tour_length(route)}")
    ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_zlim(0, NZ)
    ax.set_box_aspect((NX, NY, NZ))
    ax.view_init(elev=25, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1,2,1, projection='3d')
plot_world(ax1, init_route, "Initial tour (3‑D)")

ax2 = fig.add_subplot(1,2,2, projection='3d')
plot_world(ax2, opt_route, "Optimized tour (3‑D)")

plt.tight_layout()
plt.show()

print(f"Initial tour length : {init_len}")
print(f"Optimized tour length: {opt_len}")
print(f"Improvement          : {init_len - opt_len}")
