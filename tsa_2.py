import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

# ----------------- PARAMETERS -----------------
GRID_SIZE = 50          # size of the square grid
NUM_CITIES = 10         # number of cities
NUM_OBS   = 6           # rectangular obstacles
OBS_MAX_W = 10
OBS_MAX_H = 10
SEED      = 12          # random seed
# ------------------------------------------------

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

# ---------- build obstacle map ----------
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
obstacles = []
for _ in range(NUM_OBS):
    w = random.randint(4, OBS_MAX_W)
    h = random.randint(4, OBS_MAX_H)
    x = random.randint(0, GRID_SIZE - w - 1)
    y = random.randint(0, GRID_SIZE - h - 1)
    grid[y:y+h, x:x+w] = 1
    obstacles.append((x, y, w, h))

# ---------- A* pathfinder ----------
def astar(start, goal):
    sy, sx = start
    gy, gx = goal
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    open_heap = []
    heapq.heappush(open_heap, (abs(sy-gy)+abs(sx-gx), 0, start))
    came_from = {}
    gscore = {start: 0}
    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        cy, cx = current
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE and grid[ny, nx] == 0:
                ng = g + 1
                nbr = (ny, nx)
                if ng < gscore.get(nbr, 1e9):
                    gscore[nbr] = ng
                    priority = ng + abs(ny-gy) + abs(nx-gx)
                    heapq.heappush(open_heap, (priority, ng, nbr))
                    came_from[nbr] = current
    return None

# ---------- generate unique cities ----------
cities = []
occupied = set()
while len(cities) < NUM_CITIES:
    y = random.randint(0, GRID_SIZE-1)
    x = random.randint(0, GRID_SIZE-1)
    if grid[y,x] == 0 and (y,x) not in occupied:
        cities.append((y,x))
        occupied.add((y,x))
cities = np.array(cities)

# ---------- build distance & path matrices ----------
dist = np.zeros((NUM_CITIES, NUM_CITIES))
paths = [[None]*NUM_CITIES for _ in range(NUM_CITIES)]
for i in range(NUM_CITIES):
    for j in range(i+1, NUM_CITIES):
        path = astar(tuple(cities[i]), tuple(cities[j]))
        if path is None:
            raise RuntimeError("Pathfinding failed; try regenerating.")
        length = len(path) - 1
        dist[i,j] = dist[j,i] = length
        paths[i][j] = paths[j][i] = path

# verify no zero-length distances except diagonal
assert np.all(dist[np.triu_indices(NUM_CITIES,1)] > 0)

# ---------- initial route ----------
route = list(range(NUM_CITIES))
random.shuffle(route)

def route_len(r):
    total = 0
    for k in range(NUM_CITIES):
        a = r[k]
        b = r[(k+1)%NUM_CITIES]
        total += dist[a,b]
    return total

init_len = route_len(route)

# ---------- 2‑opt ----------
def two_opt(r):
    best = r[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, NUM_CITIES-2):
            for j in range(i+1, NUM_CITIES):
                if j-i == 1:
                    continue
                new = best[:]
                new[i:j] = best[i:j][::-1]
                if route_len(new) < route_len(best):
                    best = new
                    improved = True
        r = best
    return best

opt_route = two_opt(route)
opt_len = route_len(opt_route)

# ---------- drawing ----------
def draw(ax, r, ttl):
    ax.imshow(grid, cmap='Greys', origin='lower')
    # cities
    for idx,(y,x) in enumerate(cities):
        ax.plot(x, y, 'yo', markersize=6, markeredgecolor='black')
        ax.text(x+0.4, y+0.4, str(idx), fontsize=8, color='blue')
    # path
    for k in range(NUM_CITIES):
        a = r[k]
        b = r[(k+1)%NUM_CITIES]
        path = paths[a][b]
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=1.5)
    ax.set_title(f"{ttl}\nLength = {route_len(r)}")
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')

plt.figure(figsize=(12,6))
draw(plt.subplot(1,2,1), route, "Initial route")
draw(plt.subplot(1,2,2), opt_route, "Optimized route")
plt.tight_layout()
plt.show()

print(f"Initial length : {init_len}")
print(f"Optimized length: {opt_len}")
print(f"Improvement    : {init_len - opt_len}")
