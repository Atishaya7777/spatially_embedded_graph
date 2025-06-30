from networkx.classes import neighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
from scipy.spatial import distance_matrix


def compute_wiener_index_from_path(points, path):
    """Compute Wiener index of a path graph."""
    dist = distance_matrix(points, points)
    G = nx.Graph()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        G.add_edge(u, v, weight=dist[u, v])

    wiener = 0
    for i in range(len(points)):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='weight')
        for j in range(i + 1, len(points)):
            wiener += lengths[j]

    return wiener


def get_optimal_wiener_path(points):
    """Brute-force Hamiltonian path with minimal Wiener index."""
    n = len(points)
    best_wiener = float('inf')
    best_path = None
    for path in itertools.permutations(range(n)):
        w = compute_wiener_index_from_path(points, path)
        if w < best_wiener:
            best_wiener = w
            best_path = path
    return best_path, best_wiener


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


def generate_all_spanning_trees(G):
    n = len(G.nodes)
    all_edges = list(G.edges(data=True))

    for edges in itertools.combinations(all_edges, n - 1):
        tree = nx.Graph()

        tree.add_nodes_from(G.nodes)

        tree.add_edges_from(
            [(u, v, {'weight': float(data['weight'])}) for u, v, data in edges])

        if nx.is_connected(tree):
            yield tree


def find_best_tree(points):
    n_points = len(points)
    dist_matrix = distance_matrix(points, points)

    G_complete = nx.Graph()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            G_complete.add_edge(i, j, weight=dist_matrix[i, j])

    min_wiener = float('inf')
    best_tree = None

    for tree in generate_all_spanning_trees(G_complete):
        wiener = compute_wiener_index(tree)
        if wiener < min_wiener:
            min_wiener = wiener
            best_tree = tree.copy()

    return best_tree, min_wiener


def mst_dfs_approximation(points):
    """Approximate Hamiltonian path using MST and DFS traversal."""
    n = len(points)
    dist = distance_matrix(points, points)
    G = nx.Graph()

    # Uncomment this if you want emst
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         G.add_edge(i, j, weight=dist[i, j])

    # mst = nx.minimum_spanning_tree(G)
    mst = find_best_tree(points)[0]

    # Manual DFS
    visited = set()
    order = []
    stack = [0]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            neighbors = sorted(list(mst.neighbors(node)), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

    wiener = compute_wiener_index_from_path(points, order)
    return order, wiener, mst


def run_trial(seed, n_points):
    """Single trial for one seed."""
    np.random.seed(seed)
    points = np.random.rand(n_points, 2)
    opt_path, opt_wiener = get_optimal_wiener_path(points)
    dfs_path, dfs_wiener, opt_wiener_mst = mst_dfs_approximation(points)
    ratio = dfs_wiener / opt_wiener
    return {
        'seed': seed,
        'points': points,
        'optimal_path': opt_path,
        'optimal_wiener': opt_wiener,
        'dfs_path': dfs_path,
        'dfs_wiener': dfs_wiener,
        'approximation_ratio': ratio,
        'optimal_wiener_tree': opt_wiener_mst,
    }


def run_wiener_comparison_parallel(n_points=6, num_trials=50, max_workers=4):
    print(f"Running {num_trials} trials with {max_workers} workers...")
    seeds = list(range(num_trials))
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            run_trial, seed, n_points): seed for seed in seeds}
        for future in tqdm(as_completed(futures), total=num_trials):
            results.append(future.result())

    return sorted(results, key=lambda r: -r['approximation_ratio'])


def plot_wiener_comparison(result, save_dir, case_type):
    points = result['points']
    mst = result['optimal_wiener_tree']
    opt_path = result['optimal_path']
    dfs_path = result['dfs_path']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def plot_edges(ax, edges, color, title, linestyle='-'):
        ax.scatter(points[:, 0], points[:, 1], c='black', s=60)
        for u, v in edges:
            ax.plot(
                [points[u][0], points[v][0]],
                [points[u][1], points[v][1]],
                color=color, linewidth=2, linestyle=linestyle
            )
        for i, (x, y) in enumerate(points):
            ax.text(x + 0.02, y + 0.02, str(i), fontsize=10, weight='bold')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Subplot 1: Optimal path
    opt_edges = [(opt_path[i], opt_path[i + 1])
                 for i in range(len(opt_path) - 1)]
    plot_edges(axes[0], opt_edges, 'green',
               f"Optimal Path (W={result['optimal_wiener']:.3f})")

    # Subplot 2: DFS approximation
    dfs_edges = [(dfs_path[i], dfs_path[i + 1])
                 for i in range(len(dfs_path) - 1)]
    plot_edges(axes[1], dfs_edges, 'red',
               f"MST-DFS Approx (W={result['dfs_wiener']:.3f})")

    # Subplot 3: MST
    mst_edges = list(mst.edges())
    plot_edges(axes[2], mst_edges, 'blue', f"MST Only")

    fig.suptitle(
        f"Seed={result['seed']} | Approx. Ratio={
            result['approximation_ratio']:.3f}",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"{case_type}_seed{result['seed']}_ratio{
        result['approximation_ratio']:.3f}.png"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


def print_summary_stats(results, i, statsdir):
    ratios = [r['approximation_ratio'] for r in results]

    print(f"\nSummary Statistics over {len(results)} trials:")
    print(f"{'='*50}")
    print(f"Average Approximation Ratio: {np.mean(ratios):.3f}")
    print(f"Median Approximation Ratio:  {np.median(ratios):.3f}")
    print(f"Worst Approximation Ratio:   {np.max(ratios):.3f}")
    print(f"Best Approximation Ratio:    {np.min(ratios):.3f}")
    print(f"Standard Deviation:          {np.std(ratios):.3f}")
    print(f"{'='*50}")

    print(f"\nApproximation Ratio Distribution:")
    bins = [1.0, 1.2, 1.5, 2.0, 3.0, float('inf')]
    labels = ['1.0-1.2', '1.2-1.5', '1.5-2.0', '2.0-3.0', '3.0+']

    for i in range(len(bins) - 1):
        count = sum(1 for r in ratios if bins[i] <= r < bins[i + 1])
        percentage = count / len(ratios) * 100
        print(f"  {labels[i]}: {count:2d} trials ({percentage:5.1f}%)")

    stats_file = os.path.join(statsdir, f'wiener_summary_stats_{i}.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Summary Statistics over {len(results)} trials:\n")
        f.write("="*50 + "\n")
        f.write(f"Average Approximation Ratio: {np.mean(ratios):.3f}\n")
        f.write(f"Median Approximation Ratio:  {np.median(ratios):.3f}\n")
        f.write(f"Worst Approximation Ratio:   {np.max(ratios):.3f}\n")
        f.write(f"Best Approximation Ratio:    {np.min(ratios):.3f}\n")
        f.write(f"Standard Deviation:          {np.std(ratios):.3f}\n")
        f.write("="*50 + "\n")
        f.write("\nApproximation Ratio Distribution:\n")

        for i in range(len(bins) - 1):
            count = sum(1 for r in ratios if bins[i] <= r < bins[i + 1])
            percentage = count / len(ratios) * 100
            f.write(f"  {labels[i]}: {count:2d} trials ({percentage:5.1f}%)\n")


if __name__ == "__main__":
    n_points = 9
    num_trials = 100
    max_workers = 100

    for i in range(5, n_points + 1):
        output_dir = f"./wiener_comparison_plots/n{i}"
        os.makedirs(output_dir, exist_ok=True)

        results = run_wiener_comparison_parallel(
            i, num_trials, max_workers)
        print_summary_stats(results, i, output_dir)

        print(f"\nPlotting top 5 worst cases...")
        for j, res in enumerate(results[:5]):
            plot_wiener_comparison(res, output_dir, case_type=f"worst_{j}")

        print(f"Plotting top 3 best cases...")
        for j, res in enumerate(results[-3:]):
            plot_wiener_comparison(res, output_dir, case_type=f"best_{j}")

        print(f"\nAll plots saved to: {os.path.abspath(output_dir)}")
