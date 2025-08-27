import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
from math import dist

# Greedy bounding box divide-and-conquer with endpoint matching


def bounding_box_path(points):
    n = len(points)
    if n == 1:
        return points
    elif n == 2:
        return sorted(points, key=lambda p: p[0])
    else:
        points_sorted = sorted(points, key=lambda p: p[0])
        mid = n // 2
        left = points_sorted[:mid]
        right = points_sorted[mid:]

        left_path = bounding_box_path(left)
        right_path = bounding_box_path(right)

        pairs = [
            (left_path[0], right_path[0], True, True),
            (left_path[0], right_path[-1], True, False),
            (left_path[-1], right_path[0], False, True),
            (left_path[-1], right_path[-1], False, False),
        ]

        def connect_paths(lp, rp, rev_lp, rev_rp):
            if rev_lp:
                lp = lp[::-1]
            if rev_rp:
                rp = rp[::-1]
            return lp + rp

        min_dist = float('inf')
        best_conn = None
        for (l_end, r_end, rev_lp, rev_rp) in pairs:
            d = dist(l_end, r_end)
            if d < min_dist:
                min_dist = d
                best_conn = (rev_lp, rev_rp)

        return connect_paths(left_path, right_path, best_conn[0], best_conn[1])

# Compute Wiener index for a path


def wiener_index(path):
    return sum(dist(path[i], path[j]) for i in range(len(path)) for j in range(i + 1, len(path)))

# Brute-force optimal path (only for small n)


def optimal_path(points):
    best_w = float('inf')
    best_p = None
    for perm in permutations(points):
        w = wiener_index(perm)
        if w < best_w:
            best_w = w
            best_p = perm
    return list(best_p)

# Construct recursive hard instance


def construct_recursive_worst_case(level, offset=(0, 0)):
    if level == 0:
        return [(offset[0], offset[1])]
    shift = 4 ** level
    half = 2 ** (level - 1)

    left = construct_recursive_worst_case(level - 1, (offset[0], offset[1]))
    right = construct_recursive_worst_case(
        level - 1, (offset[0] + shift, offset[1]))
    return left + right

# Visualize two paths


def plot_paths(points, opt_path, greedy_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    def draw_path(ax, path, title):
        x, y = zip(*path)
        ax.plot(x, y, marker='o')
        for i, pt in enumerate(path):
            ax.text(pt[0], pt[1] + 1, str(i), fontsize=8, ha='center')
        ax.set_title(f"{title}\nWiener index = {wiener_index(path):.2f}")
        ax.axis('equal')

    draw_path(axs[0], opt_path, "Optimal Path")
    draw_path(axs[1], greedy_path, "Greedy Bounding Box Path")
    plt.tight_layout()
    plt.show()


# Use small level for brute-force feasibility
level = 3  # 2^3 = 8 points
points = construct_recursive_worst_case(level)
opt = optimal_path(points)
greedy = bounding_box_path(points)

plot_paths(points, opt, greedy)
