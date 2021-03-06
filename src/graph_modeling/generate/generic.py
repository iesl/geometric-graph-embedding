import networkx as nx

__all__ = [
    "convert_to_outtree",
    "remove_self_loops",
]


def convert_to_outtree(tree: nx.Graph) -> nx.DiGraph:
    """
    the graph generated by networkx.random_tree() is undirected.
    means that parent -> children and children -> parent both exist
    This function change the graph to only parent -> children

    :param tree:
    :return:
    """
    digraph = nx.DiGraph(tree)
    for u, v in nx.bfs_tree(tree, 0).edges():
        digraph.remove_edge(v, u)
    return digraph


def remove_self_loops(G):
    # TODO: G.remove_edges_from(nx.selfloop_edges(G)) ?
    for e in list(G.edges()):
        if e[0] == e[1]:
            G.remove_edge(e[0], e[1])
    return G
