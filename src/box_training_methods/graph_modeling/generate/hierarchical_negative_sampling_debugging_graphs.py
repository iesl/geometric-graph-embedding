import networkx as nx

__all__ = [
    "generate_balanced_tree",
    "generate_dag",
    "generate",
]


def generate_balanced_tree():
    edge_list = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (3, 10),
        (3, 11),
        (3, 12),
    ]
    g = nx.DiGraph()
    g.add_edges_from(edge_list)
    # g = nx.transitive_closure(g)
    return g


def generate_dag():
    edge_list = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (5, 8),
        (6, 9),
        (7, 9),
        (10, 11),
        (10, 12),
    ]
    g = nx.DiGraph()
    g.add_edges_from(edge_list)
    return g


def generate(which: str = "balanced-tree", **kwargs) -> nx.DiGraph:

    if which == "balanced-tree":
        return generate_balanced_tree()
    elif which == "dag":
        return generate_dag()
