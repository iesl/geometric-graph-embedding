import torch
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import time

from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchicalNegativeEdges



def graph_analytics(graph_npz_path, save_dir):

    graph_id = "-".join(graph_npz_path.split("/")[-3:])[:-len(".npz")]

    training_edges, num_nodes = edges_and_num_nodes_from_npz(graph_npz_path)

    HNE = HierarchicalNegativeEdges(
        edges=training_edges,
        negative_ratio=1,
        sampling_strategy="exact",
    )

    # TODO cache HNE.negative_roots



def dfs_max_depth(r, G, max_depth):
    children_max_depths = []
    print(f"r: {r}")
    print(f"\tmax depth: {max_depth}")
    for s in G.successors(r):
        children_max_depths.append(dfs_max_depth(s, G, max_depth + 1))
    print(f"\tchildren max depths: {children_max_depths}")
    return max(children_max_depths) if len(children_max_depths) > 0 else max_depth


def all_stats_to_csv(all_stats, csv_fpath):

    rows = []
    header_row = ",".join(all_stats[0].keys())
    rows.append(header_row)
    for stats in all_stats:
        values = stats.values()
        row = ",".join(values)
        rows.append(row)

    csv_str = "\n".join(rows)
    with open(csv_fpath, "w") as f:
        f.write(csv_str)


def generate_analytics_for_graphs_in_dir(graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/",
                                         save_dir="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/"):

    all_stats = []
    for root, dirs, files in os.walk(graphs_root):
        for f in files:
            if f.endswith(".npz"):
                graph_npz_path = "/".join([root, f])
                stats = graph_analytics(graph_npz_path, save_dir)
                all_stats.append(stats)

    return all_stats


if __name__ == '__main__':

    graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/kronecker_graph/a=1.0-b=0.6-c=0.5-d=0.2-log_num_nodes=12-transitive_closure=False/1619702443.npz",
                    save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/hierarchical_negative_sampling_debugging_graphs/log_num_nodes=12-transitive_closure=False-which=dag/1160028402.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/balanced_tree/branching=2-log_num_nodes=12-transitive_closure=False/2952040816.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/hierarchical_negative_sampling_debugging_graphs/log_num_nodes=12-transitive_closure=False-which=balanced-tree/1196640715.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # all_stats = generate_analytics_for_graphs_in_dir(graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/",
    #                                                  save_dir="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/")
    # all_stats_to_csv(all_stats, "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/graphs13_stats.csv")
