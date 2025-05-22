import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix


def compute_wiener_index(G):
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    '''
    The Wiener index is the sum of the distances between all pairs of vertices
    in a graph.

    W(G) = sum_{u, v in V(G)} d(u, v)

    Where d(u, v) is the weight of the shortest (minimum weight) path between
    u and v.
    '''
    return sum(lengths[u][v] for u in lengths for v in lengths[u] if u < v)


def run_experiment_general(n_points, num_trials):
    results = []

    for seed in range(num_trials):
        np.random.seed(seed)
        points = np.random.rand(n_points, 2)
        dist_matrix = distance_matrix(points, points)

        G_complete = nx.Graph()  # G_complete is a complete graph
        for i in range(n_points):
            G_complete.add_node(i)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                G_complete.add_edge(i, j, weight=dist_matrix[i, j])

        # MST
        mst = nx.minimum_spanning_tree(G_complete)
        wiener_mst = compute_wiener_index(mst)

        # Star Trees
        best_star_wiener = float('inf')
        for center in range(n_points):
            star = nx.Graph()
            star.add_nodes_from(G_complete.nodes)
            for i in range(n_points):
                if i != center:
                    star.add_edge(center, i, weight=dist_matrix[center, i])
            wiener_star = compute_wiener_index(star)
            if wiener_star < best_star_wiener:
                best_star_wiener = wiener_star

        results.append({
            'seed': seed,
            'winner': 'MST' if wiener_mst < best_star_wiener else 'Star'
        })

    mst_better = sum(1 for r in results if r['winner'] == 'MST')
    star_better = sum(1 for r in results if r['winner'] == 'Star')
    ties = num_trials - mst_better - star_better

    print(f"Out of {num_trials} random trials:")
    print(f" - MST had lower Wiener index: {mst_better} times")
    print(f" - Star had lower Wiener index: {star_better} times")
    print(f" - Ties: {ties} times")


def run_experiment_convex(n_points, num_trials):
    results = []
    star_wins = []

    for seed in range(num_trials):
        np.random.seed(seed)
        # Grab random angles from a uniform distribution in [0, 2pi]
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        # Convert angles to points on a circle of radius 1
        points = np.vstack((np.cos(angles), np.sin(angles))
                           ).T
        dist_matrix = distance_matrix(points, points)  # Eucliean distance

        G_complete = nx.Graph()
        for i in range(n_points):
            G_complete.add_node(i)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                G_complete.add_edge(i, j, weight=dist_matrix[i, j])

        # MST
        mst = nx.minimum_spanning_tree(G_complete)
        wiener_mst = compute_wiener_index(mst)

        # Star Trees
        best_star_wiener = float('inf')
        best_star = None
        best_center = None
        for center in range(n_points):
            star = nx.Graph()
            star.add_nodes_from(G_complete.nodes)
            for i in range(n_points):
                if i != center:
                    star.add_edge(center, i, weight=dist_matrix[center, i])
            wiener_star = compute_wiener_index(star)
            if wiener_star < best_star_wiener:
                best_star_wiener = wiener_star
                best_star = star
                best_center = center

        winner = 'MST' if wiener_mst < best_star_wiener else 'Star'
        results.append({
            'seed': seed,
            'winner': winner
        })

        # Store data for visualization if Star wins
        if winner == 'Star':
            star_wins.append({
                'seed': seed,
                'points': points,
                'mst': mst,
                'star': best_star,
                'trial': seed
            })

    mst_better = sum(1 for r in results if r['winner'] == 'MST')
    star_better = sum(1 for r in results if r['winner'] == 'Star')
    ties = num_trials - mst_better - star_better

    print(f"Out of {num_trials} random trials:")
    print(f" - MST had lower Wiener index: {mst_better} times")
    print(f" - Star had lower Wiener index: {star_better} times")
    print(f" - Ties: {ties} times")


run_experiment_general(6, 500)
run_experiment_convex(6, 500)
