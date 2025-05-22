import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


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


def plot_graph(points, mst, star, seed, trial, wiener_mst, wiener_star):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=100, label='Points')
    for i, (x, y) in enumerate(points):
        plt.text(x + 0.02, y, f'{i}', fontsize=12, color='black')

    mst_label_added = False
    for u, v in mst.edges():
        x = [points[u][0], points[v][0]]
        y = [points[u][1], points[v][1]]
        label = 'MST' if not mst_label_added else ""
        plt.plot(x, y, 'g-', alpha=0.5, label=label)
        mst_label_added = True

    star_label_added = False
    for u, v in star.edges():
        x = [points[u][0], points[v][0]]
        y = [points[u][1], points[v][1]]
        label = 'Star' if not star_label_added else ""
        plt.plot(x, y, 'r--', alpha=0.5, label=label)
        star_label_added = True

    plt.figtext(0.5, 0.01,
                f"Wiener Index — MST: {
                    wiener_mst:.3f} | Star: {wiener_star:.3f}",
                wrap=True, horizontalalignment='center', fontsize=12)

    plt.title(f"Trial {trial} (Seed {seed}): Star Wins")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def run_experiment(n_points, num_trials, visualize):
    results = []
    star_wins = []

    for seed in range(num_trials):
        np.random.seed(seed)
        points = np.random.rand(n_points, 2)
        dist_matrix = distance_matrix(points, points)

        # Create a complete graph so that we can run dfs to compute MST and Wiener index of a star
        G_complete = nx.Graph()
        for i in range(n_points):
            G_complete.add_node(i)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                G_complete.add_edge(i, j, weight=dist_matrix[i, j])

        mst = nx.minimum_spanning_tree(G_complete)
        wiener_mst = compute_wiener_index(mst)

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
        if winner == 'Star' and visualize:
            star_wins.append({
                'seed': seed,
                'points': points,
                'mst': mst,
                'star': best_star,
                'trial': seed,
                'wiener_mst': wiener_mst,
                'wiener_star': best_star_wiener
            })

    mst_better = sum(1 for r in results if r['winner'] == 'MST')
    star_better = sum(1 for r in results if r['winner'] == 'Star')
    ties = num_trials - mst_better - star_better

    print(f"Out of {num_trials} random trials:")
    print(f" - MST had lower Wiener index: {mst_better} times")
    print(f" - Star had lower Wiener index: {star_better} times")
    print(f" - Ties: {ties} times")

    # Visualize Star wins
    if visualize and star_wins:
        for win in star_wins:
            plot_graph(
                win['points'],
                win['mst'],
                win['star'],
                win['seed'],
                win['trial'],
                win['wiener_mst'],
                win['wiener_star']
            )
    elif visualize and not star_wins:
        print("No instances where Star had a lower Wiener index.")


run_experiment(6, 500, True)
