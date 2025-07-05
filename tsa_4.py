import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import random

# =================== CONFIG =====================
GRID = 30                 # grid size
N_OBS = 5                 # obstacle rectangles
OBS_W = 8
OBS_H = 8
SEED  = 7                 # reproducible (None for fresh)
SKIP  = 20                # save frame every SKIP expansions
MS_PER_FRAME = 90
GIF_PATH = "/mnt/data/mouse_exploration.gif"
# ================================================

# ---------- random world ----------
if SEED is not None:
    np.random.seed(SEED); random.seed(SEED)

grid = np.zeros((GRID, GRID), dtype=np.int8)  # 0 = free, 1 = obstacle
for _ in range(N_OBS):
    w = random.randint(4, OBS_W)
    h = random.randint(4, OBS_H)
    x0 = random.randint(0, GRID - w - 1)
    y0 = random.randint(0, GRID - h - 1)
    grid[y0:y0+h, x0:x0+w] = 1

# -------- start position --------
while True:
    sy, sx = random.randint(0, GRID-1), random.randint(0, GRID-1)
    if grid[sy, sx] == 0:
        break

# -------- BFS exploration --------
dirs = [(1,0),(-1,0),(0,1),(0,-1)]
visited = np.zeros_like(grid, dtype=np.int8)
visited[sy, sx] = 1
from collections import deque
q = deque([(sy, sx)])
frames = []
step = 0

def capture():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(np.where(grid==1, 0.2, 0.9), cmap='gray', vmin=0, vmax=1)
    vy, vx = np.where(visited==1)
    ax.scatter(vx, vy, s=12, c='skyblue', marker='s')
    if q:
        fy, fx = zip(*q)
        ax.scatter(fx, fy, s=18, c='lime', marker='s')
    ax.scatter([sx],[sy], c='red', s=35, marker='o')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Step {step} | explored: {visited.sum()}")
    fig.tight_layout()
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    from PIL import Image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    frames.append(Image.fromarray(img))
    plt.close(fig)

capture()

while q:
    y,x = q.popleft()
    for dy,dx in dirs:
        ny,nx = y+dy, x+dx
        if 0<=ny<GRID and 0<=nx<GRID and grid[ny,nx]==0 and visited[ny,nx]==0:
            visited[ny,nx]=1
            q.append((ny,nx))
    step += 1
    if step % SKIP == 0:
        capture()

capture()  # final

# save GIF
frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=MS_PER_FRAME, loop=0)

print("GIF saved at:", GIF_PATH)


