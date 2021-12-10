import networkx as nx
from loguru import logger

try:
    import graph_tool as gt
    from graph_tool.generation import price_network
except ImportError as e:
    logger.warning(
        "Could not import graph_tool, did you install it? (conda install -c conda-forge graph-tool)"
    )
    raise e


def generate(
    log_num_nodes: int, seed: int, m: int, c: float, gamma: float, **kwargs
) -> nx.DiGraph:
    gt.seed_rng(seed)
    num_nodes = 2 ** log_num_nodes
    g = price_network(num_nodes, m, c, gamma)
    ngx = nx.DiGraph()
    for s, t in g.iter_edges():
        ngx.add_edge(t, s)
    return ngx
