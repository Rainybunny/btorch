from matplotlib.axes import Axes


def plot_network(sparse_mat, ax: Axes):
    import networkx

    G = networkx.from_scipy_sparse_matrix(sparse_mat)
    pos = networkx.spring_layout(G)
    networkx.draw(
        G, pos, with_labels=True, node_color="skyblue", edge_color="gray", ax=ax
    )
