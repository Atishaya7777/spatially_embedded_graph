import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations


def generate_recursive_points(n):
    if n == 2:
        return [(0, 0), (1, 0)]
    else:
        half_n = n // 2
        left_points = generate_recursive_points(half_n)
        D_n = 4 * n
        right_points = [(x + D_n, y) for (x, y) in left_points]
        return left_points + right_points


def pairwise_dist(a, b):
    # Manhattan, but points all on y=0 so same as Euclidean x-dist
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def wiener_index(path):
    n = len(path)
    edges = [pairwise_dist(path[i], path[i+1]) for i in range(n-1)]
    prefix_sum = [0]
    for e in edges:
        prefix_sum.append(prefix_sum[-1] + e)

    total = 0
    for i in range(n):
        for j in range(i+1, n):
            total += prefix_sum[j] - prefix_sum[i]
    return total


def brute_force_wiener_path(points):
    min_wiener = float('inf')
    best_path = None
    for perm in permutations(points):
        w = wiener_index(perm)
        if w < min_wiener:
            min_wiener = w
            best_path = perm
    return best_path, min_wiener


def bounding_box_path_greedy(points):
    if len(points) <= 2:
        return sorted(points, key=lambda p: p[0])

    points_sorted = sorted(points, key=lambda p: p[0])
    mid = len(points_sorted) // 2
    left = points_sorted[:mid]
    right = points_sorted[mid:]

    left_path = bounding_box_path_greedy(left)
    right_path = bounding_box_path_greedy(right)

    def connect_paths(lp, rp, rev_lp, rev_rp):
        if rev_lp:
            lp = lp[::-1]
        if rev_rp:
            rp = rp[::-1]
        return lp + rp

    # Try all 4 combinations of endpoint connections (reversing left/right subpaths)
    candidates = [
        (False, False),
        (False, True),
        (True, False),
        (True, True)
    ]

    min_dist = float('inf')
    best_path = None
    for rev_lp, rev_rp in candidates:
        lp_mod = left_path[::-1] if rev_lp else left_path
        rp_mod = right_path[::-1] if rev_rp else right_path
        dist = pairwise_dist(lp_mod[-1], rp_mod[0])
        if dist < min_dist:
            min_dist = dist
            best_path = lp_mod + rp_mod

    return best_path


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
            dist = pairwise_dist(l_end, r_end)
            if dist < min_dist:
                min_dist = dist
                best_conn = (rev_lp, rev_rp)

        return connect_paths(left_path, right_path, best_conn[0], best_conn[1])


def plot_points(points, ax=None):
    x_vals, y_vals = zip(*points)
    if ax is None:
        plt.figure(figsize=(10, 1))
        ax = plt.gca()
    ax.scatter(x_vals, y_vals, color='blue', zorder=5)
    ax.set_yticks([])
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(-1, 1)
    return ax


def plot_points_with_vertical_offset(path, color='blue', label=None, ax=None, offset=0.1):
    x_vals = [p[0] for p in path]
    y_vals = [i * offset for i in range(len(path))]

    if ax is None:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

    ax.scatter(x_vals, y_vals, color=color, label=label, zorder=5)
    for i in range(len(path) - 1):
        ax.plot(x_vals[i:i+2], y_vals[i:i+2],
                color=color, linewidth=2, zorder=10)

    ax.set_yticks([])
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(-offset, y_vals[-1] + offset)
    if label is not None:
        ax.legend()
    return ax


def plot_path(path, color='red', label=None, ax=None):
    x_vals, y_vals = zip(*path)
    if ax is None:
        plt.figure(figsize=(10, 1))
        ax = plt.gca()
    ax.plot(x_vals, y_vals, color=color, label=label, linewidth=2, zorder=10)


def main():
    n = 8  # Brute force feasible up to 8 points
    points = generate_recursive_points(n)

    # Compute optimal path by brute force
    optimal_path, optimal_wiener = brute_force_wiener_path(points)
    print(f"Optimal Wiener index: {optimal_wiener}")

    # Compute bounding box heuristic path
    bb_path = bounding_box_path_greedy(points)
    bb_wiener = wiener_index(bb_path)
    print(f"Bounding box Wiener index: {bb_wiener}")

    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Optimal path, normal plot
    axs[0, 0].set_title("Optimal Path (Normal)")
    plot_points(points, axs[0, 0])
    plot_path(optimal_path, color='green', label='Optimal Path', ax=axs[0, 0])
    axs[0, 0].legend()

    # Optimal path, vertical offset plot
    axs[0, 1].set_title("Optimal Path (Vertical Offset)")
    plot_points_with_vertical_offset(optimal_path, color='green',
                                     label='Optimal Path', ax=axs[0, 1])

    # Bounding box path, normal plot
    axs[1, 0].set_title("Bounding Box Path (Normal)")
    plot_points(points, axs[1, 0])
    plot_path(bb_path, color='red', label='Bounding Box Path', ax=axs[1, 0])
    axs[1, 0].legend()

    # Bounding box path, vertical offset plot
    axs[1, 1].set_title("Bounding Box Path (Vertical Offset)")
    plot_points_with_vertical_offset(bb_path, color='red',
                                     label='Bounding Box Path', ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
