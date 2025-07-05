# 🧭 Path Explorer — Visual DFS Simulation

A lightweight Python-based visualizer simulating how a depth-first search (DFS) algorithm explores a 2D grid world filled with obstacles — inspired by how a mouse might explore a maze.

This project uses **matplotlib** for rendering real-time exploration steps.

---

## 📌 Features

* ✅ **DFS (Depth-First Search)** based path exploration
* 🧱 **Random obstacle placement** using rectangles
* 🎨 **Color-coded visualization**:

  * **White**: unexplored free space
  * **Black**: obstacles
  * **Sky blue**: visited cells
  * **Green**: current DFS path
  * **Red**: current search head
* 📐 30×30 customizable grid
* 💡 Real-time simulation using `matplotlib` (no GUI dependencies like `tkinter`)

---

## 🚀 Getting Started

### 📦 Requirements

* Python 3.7+
* `matplotlib`
* `numpy`

### 🔧 Installation

```bash
pip install matplotlib numpy
```

### ▶️ Run

```bash
python path_explorer_visual.py
```

A live animation window will open and show the exploration progress.

---

## ⚙️ Configuration

You can tweak the following values in the source code:

```python
GRID_SIZE = 30        # grid size
STEP_DELAY = 0.05     # seconds per frame
OBSTACLE_COUNT = 10   # number of rectangular obstacles
RNG_SEED = 42         # seed for reproducible randomness
```

---

## 🧠 Algorithm

The simulation uses a **clockwise DFS traversal**:

* Direction order: North → East → South → West
* Explores new cells if they're within bounds, free of obstacles, and unvisited
* If stuck, backtracks along the DFS path
* Ends when all reachable cells have been visited

---

## 📂 File Structure

```
├── path_explorer_visual.py   # Main visualisation script
├── README.md                 # Project documentation
```

---

## 🏁 Planned Features

*

---

## 📜 License

MIT License © 2025 Atharva M

---

## 🙋‍♂️ Contributions & Feedback

Feel free to fork, raise issues, or suggest features!

---

Happy Exploring 🐭

