import torch
import os
import numpy as np
import networkx as nx
import json
import time
import pickle

from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchicalNegativeEdges


def cache_hns(graph_npz_path, graph_hns_dir):

    training_edges, num_nodes = edges_and_num_nodes_from_npz(graph_npz_path)
    HNE = HierarchicalNegativeEdges(
        edges=training_edges,
        negative_ratio=1,  # doesn't matter for caching
        sampling_strategy="exact",
    )
    torch.save(HNE.negative_roots, os.path.join(graph_hns_dir, "negative_roots.pt"))

    node_to_num_descendants = {n: len(nx.descendants(HNE.G, n)) for n in HNE.G.nodes}
    with open(os.path.join(graph_hns_dir, "node_to_num_descendants.pkl"), 'wb') as f:
        pickle.dump(node_to_num_descendants, f, protocol=pickle.HIGHEST_PROTOCOL)


def traverse_and_cache_hns(graphs_dir, graph_types=['price']): # 'balanced_tree', 'nested_chinese_restaurant_process', 'price']):

    for graph_type in graph_types:
        graph_root = os.path.join(graphs_dir, graph_type)
        for root, dirs, files in os.walk(graph_root):
            for f in files:
                if f.endswith(".npz"):
                    graph_npz_path = "/".join([root, f])
                    graph_hns_dir = graph_npz_path[:-len(".npz")] + ".hns"
                    try:
                        os.mkdir(graph_hns_dir)
                        cache_hns(graph_npz_path, graph_hns_dir)
                    except FileExistsError:
                        pass


if __name__ == '__main__':
    # traverse_and_cache_hns("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs")
    traverse_and_cache_hns("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/")
