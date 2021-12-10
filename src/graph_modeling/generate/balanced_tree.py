import networkx as nx

from graph_modeling.generate.generic import convert_to_outtree

__all__ = [
    "generate",
]


def generate(log_num_nodes: int, branching: int, **kwargs) -> nx.DiGraph:
    num_nodes = 2 ** log_num_nodes
    height = 0
    count_nodes = 0
    while count_nodes < num_nodes:
        count_nodes += branching ** height
        height += 1

    height -= 1

    tree = nx.balanced_tree(branching, height)
    tree.remove_nodes_from(list(range(num_nodes, count_nodes)))
    tree = convert_to_outtree(tree)
    return tree
