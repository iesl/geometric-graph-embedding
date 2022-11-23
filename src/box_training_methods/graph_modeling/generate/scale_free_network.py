import networkx as nx

from .generic import remove_self_loops

__all__ = [
    "generate",
]


def generate(
    log_num_nodes: int,
    alpha: float,
    gamma: float,
    delta_in: float,
    delta_out: float,
    seed: int,
    **kwargs
) -> nx.DiGraph:
    num_nodes = 2 ** log_num_nodes
    g = nx.scale_free_graph(
        num_nodes,
        alpha=alpha,
        beta=1 - alpha - gamma,
        gamma=gamma,
        delta_in=delta_in,
        delta_out=delta_out,
        seed=seed,
    )
    g = nx.DiGraph(g)
    g = remove_self_loops(g)
    return g
