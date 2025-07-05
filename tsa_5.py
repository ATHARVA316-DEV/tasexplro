import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# =================== CONFIG =====================
GRID = 30                 # grid size
N_OBS = 5                 # obstacle rectangles
OBS_W = 8
OBS_H = 8
SEED  = 7                 # reproducible (None for fresh)
SKIP  = 20                # save frame every SKIP moves
MS_PER_FRAME = 90
GIF_PATH = "/mnt/data/mouse_exploration_dfs.gif"
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

# -------- DFS exploration (one path at a time) --------
dirs = [(0,-1), (1,0), (0,1), (-1,0)]  # N, E, S, W (clockwise)
visited = np.zeros_like(grid, dtype=np.int8)
visited[sy, sx] = 1

# Stack entries: (y, x, dir_index) where dir_index = next direction to try
stack = [(sy, sx, 0)]
frames = []
moves = 0

def capture(step):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(np.where(grid==1, 0.2, 0.9), cmap='gray', vmin=0, vmax=1)
    vy, vx = np.where(visited==1)
    ax.scatter(vx, vy, s=12, c='skyblue', marker='s', label='Visited')
    
    # current path (stack)
    if stack:
        py, px = zip(*[(s[1], s[0]) for s in stack])  # x first for scatter
        ax.scatter(py, px, s=16, c='lime', marker='s', label='Path')
        cy, cx, _ = stack[-1]
        ax.scatter([cx], [cy], s=40, c='red', marker='o', label='Head')
    
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Move {step} | visited: {visited.sum()}")
    ax.legend(loc='upper right', fontsize='x-small', framealpha=0.7)
    fig.tight_layout()
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    frames.append(Image.fromarray(img))
    plt.close(fig)

capture(moves)  # initial frame

while stack:
    y, x, d_idx = stack[-1]
    found = False
    # try directions starting from d_idx
    for i in range(d_idx, 4):
        dy, dx = dirs[i]
        ny, nx = y+dy, x+dx
        if 0<=ny<GRID and 0<=nx<GRID and grid[ny,nx]==0 and visited[ny,nx]==0:
            # update current node's next direction to try later
            stack[-1] = (y, x, i+1)
            # push new node
            visited[ny,nx] = 1
            stack.append((ny, nx, 0))
            found = True
            moves += 1
            if moves % SKIP == 0:
                capture(moves)
            break
    if not found:
        # backtrack
        stack.pop()
        moves += 1
        if moves % SKIP == 0:
            capture(moves)

# final frame
capture(moves)

# save GIF
frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=MS_PER_FRAME, loop=0)

print("DFS exploration GIF saved at:", GIF_PATH)
