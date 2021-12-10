import random

import networkx as nx
from loguru import logger
from typing import *

__all__ = [
    "generate",
]


def generate(log_num_nodes: int, seed: int, alpha: int, **kwargs) -> nx.DiGraph:
    """
    Generate a nCRP graph with `num_nodes` nodes and parameter `alpha` which represents the
    "number of people you expect to be sitting at a new table"
    """
    num_nodes = 2 ** log_num_nodes
    nodes_to_process = [(0, num_nodes, None)]  # id, num nodes, parent
    edges = []
    rng = random.Random(seed)
    next_id = 1
    while nodes_to_process:
        logger.debug(f"num nodes to process: {len(nodes_to_process)}")
        node_id, count, parent = nodes_to_process.pop(0)
        if parent is not None:
            edges.append((parent, node_id))
        if count > 1:
            counts_of_kids = sample_crp(count, alpha, rng)
            for kid_count in counts_of_kids:
                nodes_to_process.append((next_id, kid_count, node_id))
                next_id += 1

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def sample_crp(N: int, alpha: float, rng: random.Random) -> List[int]:
    """
    Return a list of integers representing the number of people sitting at each table in a Chinese restaurant process.
    :param N: number of people to seat
    :param alpha: number of people you expect to be sitting at a new table
    :param rng: random number generator
        (provided as an argument, as we will call this repeatedly and want to seed it prior)

    returns: List of integers, where the ith integer represents the number of people seated at the ith table
    """
    # begin with one "imagined" table
    people_per_table = [alpha]
    for i in range(N):
        # sample a table to sit at
        x = rng.random() * (i + alpha)
        sampled_table = 0
        for people_at_this_table in people_per_table:
            x -= people_at_this_table
            if x < 0:
                break
            sampled_table += 1

        if sampled_table == len(people_per_table) - 1:
            # keep the new table, which actually only has 1 person
            people_per_table[-1] = 1
            # set up a new one
            people_per_table.append(alpha)
        else:
            # otherwise, simply increment the count on existing table
            people_per_table[sampled_table] += 1
            # and leave that new table set up
    # return all but the last "imagined" table
    return people_per_table[:-1]
