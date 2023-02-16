import torch
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchicalNegativeEdges


def save_histogram(negatives_per_node, random_or_hierarchical, graph_id, save_dir):

    plt.hist(np.bincount(negatives_per_node), bins=np.arange(min(negatives_per_node).item(), max(negatives_per_node).item()))
    plt.title(f"{graph_id} {random_or_hierarchical}")
    plt.xlabel("# negatives")
    plt.ylabel("# nodes")
    filename = "/".join([save_dir, f"{graph_id}_{random_or_hierarchical}"]) + ".png"
    plt.savefig(filename)
    plt.clf()


def graph_analytics(graph_npz_path, save_dir):

    graph_id = "-".join(graph_npz_path.split("/")[-3:])[:-len(".npz")]

    training_edges, num_nodes = edges_and_num_nodes_from_npz(graph_npz_path)

    HNE = HierarchicalNegativeEdges(
        edges=training_edges,
        negative_ratio=16,
        sampling_strategy="exact",
    )

    G = HNE.G
    density = nx.density(G)

    # RANDOM STATS
    num_rand_negatives_per_node = torch.tensor((num_nodes - HNE.A.sum(axis=0))).squeeze()       # everybody but parents is a possible random negative parent
    max_num_rand_negatives = torch.max(num_rand_negatives_per_node).item()
    min_num_rand_negatives = torch.min(num_rand_negatives_per_node).item()
    avg_num_rand_negatives = torch.mean(num_rand_negatives_per_node.float()).item()
    # save_histogram(negatives_per_node=num_rand_negatives_per_node,
    #                random_or_hierarchical="random",
    #                graph_id=graph_id,
    #                save_dir=save_dir)

    # HIERARCHICAL STATS
    num_hier_negative_roots_per_node = (HNE.negative_roots != HNE.EMB_PAD).int().sum(dim=-1)  # TODO save in file as histogram (wandb or run dir)
    max_num_hier_negative_roots = HNE.negative_roots.shape[-1]
    min_num_hier_negative_roots = torch.min(num_hier_negative_roots_per_node).item()
    avg_num_hier_negative_roots = torch.mean(num_hier_negative_roots_per_node.float()).item()
    # save_histogram(negatives_per_node=num_hier_negative_roots_per_node,
    #                random_or_hierarchical="hierarchical",
    #                graph_id=graph_id,
    #                save_dir=save_dir)

    # the greater this is the more efficient hierarchical sampling will be
    avg_rand_to_avg_hier_ratio = avg_num_rand_negatives / avg_num_hier_negative_roots

    stats = {
        "graph_id": graph_id,
        "graph_density": density,
        "# nodes": num_nodes,
        "random": {
            "max # negatives": max_num_rand_negatives,
            "min # negatives": min_num_rand_negatives,
            "avg # negatives": avg_num_rand_negatives,
        },
        "hierarchical": {
            "max # negative roots": max_num_hier_negative_roots,
            "min # negative roots": min_num_hier_negative_roots,
            "avg # negative roots": avg_num_hier_negative_roots,
        },
        "avg random to avg hierarchical ratio": avg_rand_to_avg_hier_ratio,
    }

    with open("/".join([save_dir, graph_id]) + ".json", "w") as f:
        json.dump(stats, f, sort_keys=False, indent=4)

    return stats


def generate_analytics_for_graphs_in_dir(graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/",
                                         save_dir="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/"):

    for root, dirs, files in os.walk(graphs_root):
        for f in files:
            if f.endswith(".npz"):
                graph_npz_path = "/".join([root, f])
                graph_analytics(graph_npz_path, save_dir)

if __name__ == '__main__':

    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/nested_chinese_restaurant_process/alpha=10-log_num_nodes=12-transitive_closure=False/333283769.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/hierarchical_negative_sampling_debugging_graphs/log_num_nodes=12-transitive_closure=False-which=dag/1160028402.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/balanced_tree/branching=2-log_num_nodes=12-transitive_closure=False/2952040816.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/hierarchical_negative_sampling_debugging_graphs/log_num_nodes=12-transitive_closure=False-which=balanced-tree/1196640715.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    generate_analytics_for_graphs_in_dir(graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/",
                                         save_dir="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/")
