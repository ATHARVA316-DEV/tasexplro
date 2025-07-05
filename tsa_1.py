import numpy as np
import matplotlib.pyplot as plt
import random
from math import hypot

# ----- CONFIGURATION -----
num_cities = 15         # number of cities in the simulation
seed = 42               # random seed for reproducibility (set to None for different every run)
# -------------------------

if seed is not None:
    np.random.seed(seed)
    random.seed(seed)

# Generate random 2‑D coordinates for each city
coords = np.random.rand(num_cities, 2)

# Helper: compute the total length of a closed tour
def total_distance(route):
    d = 0.0
    for i in range(len(route)):
        x1, y1 = coords[route[i]]
        x2, y2 = coords[route[(i + 1) % len(route)]]
        d += hypot(x2 - x1, y2 - y1)
    return d

# Start with a random route
initial_route = list(range(num_cities))
random.shuffle(initial_route)
init_len = total_distance(initial_route)

# ------- 2‑opt local search for improvement -------
def two_opt(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue  # edges share a vertex; skip
                new_route = best[:]
                new_route[i:j] = best[i:j][::-1]  # 2‑opt swap
                if total_distance(new_route) < total_distance(best):
                    best = new_route
                    improved = True
        route = best
    return best

optimized_route = two_opt(initial_route)
opt_len = total_distance(optimized_route)

# ------------- Visualization ----------------
# Plot the initial tour
fig1 = plt.figure()
plt.scatter(coords[:, 0], coords[:, 1])
for idx, city in enumerate(initial_route):
    next_city = initial_route[(idx + 1) % num_cities]
    x1, y1 = coords[city]
    x2, y2 = coords[next_city]
    plt.plot([x1, x2], [y1, y2])
    plt.annotate(str(city), (x1, y1), textcoords="offset points", xytext=(5, 5))
plt.title(f"Initial tour — length = {init_len:.3f}")
plt.axis("equal")
plt.show()

# Plot the optimized tour
fig2 = plt.figure()
plt.scatter(coords[:, 0], coords[:, 1])
for idx, city in enumerate(optimized_route):
    next_city = optimized_route[(idx + 1) % num_cities]
    x1, y1 = coords[city]
    x2, y2 = coords[next_city]
    plt.plot([x1, x2], [y1, y2])
    plt.annotate(str(city), (x1, y1), textcoords="offset points", xytext=(5, 5))
plt.title(f"Optimized tour — length = {opt_len:.3f}")
plt.axis("equal")
plt.show()

print(f"Initial distance:   {init_len:.3f}")
print(f"Optimized distance: {opt_len:.3f}")
print(f"Improvement:        {init_len - opt_len:.3f}")
