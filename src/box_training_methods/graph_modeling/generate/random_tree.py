import networkx as nx

from .generic import convert_to_outtree

__all__ = [
    "generate",
]


def generate(log_num_nodes: int, seed: int, **kwargs) -> nx.DiGraph:
    num_nodes = 2 ** log_num_nodes
    t = nx.random_tree(n=num_nodes, seed=seed)
    t = convert_to_outtree(t)
    return t
