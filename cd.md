from pathlib import Path, PurePosixPath

code = """import random
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
GRID_SIZE = 30
STEP_DELAY = 0.05
OBSTACLE_COUNT = 10
RNG_SEED = 42

COLORS = {
    "free": [1.0, 1.0, 1.0],       # white
    "obstacle": [0.0, 0.0, 0.0],   # black
    "visited": [0.75, 0.9, 1.0],   # light blue
    "path": [0.2, 1.0, 0.2],       # green
    "head": [1.0, 0.0, 0.0],       # red
}

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W


class World:
    def __init__(self, obstacle_count: int, rng: random.Random):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.visited = np.zeros_like(self.grid, dtype=bool)
        self.rng = rng
        self._place_obstacles(obstacle_count)
        self.start = self._find_start()
        self.visited[self.start] = True
        self.stack = [(*self.start, 0)]
        self.steps = 0

    def _place_obstacles(self, count: int):
        for _ in range(count):
            w = self.rng.randint(4, 8)
            h = self.rng.randint(4, 8)
            x0 = self.rng.randint(0, GRID_SIZE - w)
            y0 = self.rng.randint(0, GRID_SIZE - h)
            self.grid[y0:y0+h, x0:x0+w] = 1

    def _find_start(self) -> Tuple[int, int]:
        while True:
            y = self.rng.randint(0, GRID_SIZE - 1)
            x = self.rng.randint(0, GRID_SIZE - 1)
            if self.grid[y, x] == 0:
                return y, x

    def step(self):
        if not self.stack:
            return
        y, x, d = self.stack[-1]
        for i in range(d, 4):
            dy, dx = DIRECTIONS[i]
            ny, nx = y + dy, x + dx
            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                if self.grid[ny, nx] == 0 and not self.visited[ny, nx]:
                    self.stack[-1] = (y, x, i + 1)
                    self.visited[ny, nx] = True
                    self.stack.append((ny, nx, 0))
                    self.steps += 1
                    return
        self.stack.pop()
        self.steps += 1

    def finished(self):
        return not self.stack

    def render(self) -> np.ndarray:
        img = np.tile(COLORS["free"], (GRID_SIZE, GRID_SIZE, 1))
        img[self.grid == 1] = COLORS["obstacle"]
        img[self.visited] = COLORS["visited"]
        for y, x, _ in self.stack:
            img[y, x] = COLORS["path"]
        if self.stack:
            hy, hx, _ = self.stack[-1]
            img[hy, hx] = COLORS["head"]
        return img


# ---------- MAIN ----------
def main():
    rng = random.Random(RNG_SEED)
    world = World(OBSTACLE_COUNT, rng)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(world.render(), origin="upper")
    title = ax.set_title("DFS Path Explorer — Step 0")

    while not world.finished():
        world.step()
        im.set_data(world.render())
        title.set_text(f"DFS Path Explorer — Step {world.steps}")
        fig.canvas.draw()
        plt.pause(STEP_DELAY)

    plt.ioff()
    title.set_text(f"Exploration Complete — {world.steps} steps ✔️")
    plt.show()


if __name__ == "__main__":
    main()
"""

content = f"""# Path‑Explorer — Matplotlib Visual Version
_A self‑contained DFS explorer visualiser in Python using Matplotlib._

---

## Overview

This script visualises a **Depth‑First Search (DFS)** across a 30×30 grid populated with random rectangular obstacles. Each frame shows:

- **Free cells** – white  
- **Obstacles** – black  
- **Visited cells** – light blue  
- **Current DFS path** – green  
- **Explorer “head”** – red  

The algorithm proceeds one DFS step at a time so you can watch exactly how the search backtracks and eventually covers the whole reachable space.

---

## 🎛️ Configuration

| Constant | Default | What it does |
| -------- | ------- | ------------ |
| `GRID_SIZE` | `30` | Width and height of the square grid. |
| `OBSTACLE_COUNT` | `10` | How many rectangular obstacles to drop. |
| `RNG_SEED` | `42` | Seed for repeatable random worlds. |
| `STEP_DELAY` | `0.05` | Seconds to pause between frames. |

Feel free to tweak these at the top of the script.

---

## 📦 Requirements

```bash
pip install numpy matplotlib"""