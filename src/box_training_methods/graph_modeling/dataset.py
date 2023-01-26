import math
from pathlib import Path
from time import time
from typing import *

import attr
import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.sparse import load_npz, csr_matrix
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, WeightedRandomSampler

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

import networkx as nx

from ..enums import PermutationOption

__all__ = [
    "node_counts",
    "parent_child_node_counts",
    "edges_from_tsv",
    "edges_from_pos_neg_hec_tsv",
    "edges_and_num_nodes_from_npz",
    "convert_edges_to_ints",
    "convert_ints_to_edges",
    "RandomEdges",
    "RandomNegativeEdges",
    "HierarchicalNegativeEdgesDebug",
    "HierarchicalNegativeEdgesBatched",
    "GraphDataset",
]


def node_counts(indices: LongTensor, num_nodes: int) -> LongTensor:
    """
    Count the number of times each node appears in indices.

    :param indices: LongTensor of node indices
    :param num_nodes: number of nodes
    :return: count vector `x` such that `x[i]` is the number of times `i` appears in `indices`
    """
    unique_indices, nonzero_counts = torch.unique(indices, return_counts=True)
    counts = torch.zeros(num_nodes, dtype=torch.long, device=indices.device)
    counts[unique_indices] = nonzero_counts
    return counts


def parent_child_node_counts(edges: LongTensor, num_nodes: int) -> LongTensor:
    """
    Separately count the number of times each node appears as a parent and as a child.

    :param edges: Edges represented as a LongTensor with shape (..., 2), where [...,0] is
        parent and [..., 1] is child.
    :param num_nodes: integer representing total number of nodes
    :return: LongTensor of shape (num_nodes, 2) where [i,0] is the number of times i appears
        as a parent and [i,1] the number of times i appears as a child
    """
    parent_count = node_counts(edges[..., 0], num_nodes)
    child_count = node_counts(edges[..., 1], num_nodes)
    return torch.stack((parent_count, child_count), dim=-1)


def edges_from_tsv(edge_file: Union[Path, str]) -> LongTensor:
    """
    Loads edges from a given tsv file into a PyTorch LongTensor.
    Meant for importing data where each edge appears as a line in the file, with
        <parent_id>\t<child_id>

    :param edge_file: Path of edge tsv file in format above
    :returns: PyTorch LongTensor of edges with shape (num_edges, 2)
    """
    start = time()
    logger.info(f"Loading edges from {edge_file}...")
    edges = torch.tensor(pd.read_csv(edge_file, sep="\t", header=None).to_numpy())
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges


def edges_from_pos_neg_hec_tsv(
    pos_edge_path: Union[Path, str], neg_edge_path: Union[Path, str], num_neg: int = 10
) -> LongTensor:
    """
    Loads separate positive and negative tsv files into a single PyTorch LongTensor
    Meant for importing data in the format used by Hyperbolic Entailment Cones, where each line in the positive
    tsv file corresponds to 10 lines of the negatives file.

    :param pos_edge_path: Path of positive edge tsv file
    :param neg_edge_path: Path of positive edge tsv file
    :param num_neg: Number of negatives for each positive
        Note: this shouldn't be *decided* in this function, it is a property of the data. If this number is 10, for
        example, then lines 1-10 of neg_edge_path will be associated with the positive edge on line 1 of pos_edge_path.
    :returns: Pytorch LongTensor of edges with shape (num_pos_edges, 1 + neg_ratio, 2), where [i,0,:] is the positive
        edge and [i,1:,:] are the (associated, permuted) negatives
    """
    pos_edge_path = edges_from_tsv(pos_edge_path)[:, None, :]
    neg_edge_path = edges_from_tsv(neg_edge_path).view(-1, num_neg, 2)
    return torch.cat((pos_edge_path, neg_edge_path), dim=1)


def edges_and_num_nodes_from_npz(path: Union[str, Path]) -> Tuple[LongTensor, int]:
    """
    Loads edges and number of nodes from an npz file. Meant for importing synthetic graph data.

    :param path: Location of npz file.
    :returns: Pytorch LongTensor of edges and int representing number of nodes
    """
    start = time()
    logger.info(f"Loading {path}...")
    digraph_coo = load_npz(path)
    logger.info(f"Creating PyTorch LongTensor representation of edges...")
    out_node_list = digraph_coo.row
    in_node_list = digraph_coo.col
    edges = torch.from_numpy(np.stack((out_node_list, in_node_list), axis=-1)).long()
    num_nodes = digraph_coo.shape[0]
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges, num_nodes


def _validate_edge_tensor(instance, attribute, value) -> None:
    """
    Validator that ensures edges tensor is a LongTensor with shape (..., 2)
    """
    if not isinstance(value, Tensor):
        raise ValueError(f"Edges should be a LongTensor, but received {type(value)}")
    if value.dtype != torch.long:
        raise ValueError(
            f"Edges should have dtype=torch.long, but received edges with dtype={value.dtype}"
        )
    if value.shape[-1] != 2:
        raise ValueError(
            f"Edges should have shape (..., 2), but received edges with shape {value.shape}"
        )


def _node_probability_converter(
    node_probabilities: Optional[Tensor],
) -> Optional[Tensor]:
    """
    Convert node probabilities to the proper shape and type for use with multinomial.

    :param node_probabilities: Tensor of shape (num_nodes) or (num_nodes, 2)
    :return: transposed and expanded form of node_probabilities with shape (2, num_nodes),
        which can be used with torch.multinomial
    """
    if node_probabilities is None:
        return None
    original_shape = node_probabilities.shape
    if len(node_probabilities.shape) == 1:
        node_probabilities = node_probabilities[:, None].expand(-1, 2)
    if len(node_probabilities.shape) != 2 or node_probabilities.shape[1] != 2:
        raise ValueError(
            f"node_probabilities.shape={original_shape} should have shape (num_nodes,) or (num_nodes, 2)"
        )
    return node_probabilities.transpose(0, 1).float()


def convert_edges_to_ints(edges: LongTensor, num_nodes: int) -> LongTensor:
    """
    Convert edges from (i, j) form to integers k = i*num_nodes + j.

    :param edges: LongTensor with shape (..., 2) where edges[...,0] is the tail and edges[...,1] is the head.
    :param num_nodes: number of nodes
    :returns: LongTensor with shape (...,) where the value is an integer representation of an edge
    """
    return edges[..., 0] * num_nodes + edges[..., 1]


def convert_ints_to_edges(ints: LongTensor, num_nodes: int) -> LongTensor:
    """
    Convert integers k to edges (k // num_nodes, k % num_nodes)

    :param ints: LongTensor with shape (...,) where the value is an integer representation of an edge
    :param num_nodes: number of nodes
    :returns: LongTensor with shape (..., 2) where [...,0] is the node id of tail and [...,1] is the node id of head.
    """
    return torch.stack(
        (torch.div(ints, num_nodes, rounding_mode="trunc"), ints % num_nodes), dim=-1
    )


@attr.s(auto_attribs=True)
class RandomEdges:
    """
    Return randomly sampled edges, using various probabilities.
    Note: These are simply random edges, not random *negative* edges. This class makes no attempt to verify that these
    edges do not appear in the graph.

    This uses no additional RAM, apart from any tensors provided as input.

    :param num_nodes: Number of nodes in the graph
    :param negative_ratio: Number of negatives to sample for each positive
    :param permuted: If True, for each positive edge (i,j) return negative edges (i', j) or (j, i').
    :param permuted_probabilities: If provided, is used to determine if (i,j) should be permuted to (i', j) or (j', i).
    :param node_probabilities: If provided, will sample new nodes according to this tensor of weights.
        Should have shape (num_nodes,) or (num_nodes, 2), where the latter indicates different weights for tail/head;
            node_probabilities[i,0] = probability to sample i as tail node
            node_probabilities[i,1] = probability to sample i as head node
    """

    num_nodes: int
    negative_ratio: int = 10
    permuted: bool = False
    permuted_probabilities: Optional[Tensor] = None
    node_probabilities: Optional[Tensor] = attr.ib(
        default=None, converter=_node_probability_converter
    )
    avoid_edges: Optional[LongTensor] = None

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2)
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """

        negative_shape = (*positive_edges.shape[:-1], self.negative_ratio, 2)

        # sample negatives
        if self.node_probabilities is None:
            negative_edges = torch.randint(
                self.num_nodes, negative_shape, device=positive_edges.device
            )
        else:
            num_samples = math.prod(negative_shape[:-1])
            negative_edges = (
                torch.multinomial(
                    self.node_probabilities, num_samples, replacement=True,
                )
                .transpose(0, 1)
                .reshape(negative_shape)
                .to(positive_edges.device)
            )

        if self.permuted:
            if self.permuted_probabilities is None:
                permuted_probabilities = 0.5 * torch.ones(negative_shape[:-1])
            else:
                raise NotImplementedError(
                    "Using non-uniform permuted probabilities is not supported yet"
                )
            negative_node_mask = (torch.bernoulli(permuted_probabilities) == 1).to(
                negative_edges.device
            )
            # a 0 means (i',j), a 1 means (i,j')
            negative_node_mask = torch.stack(
                (negative_node_mask, ~negative_node_mask), dim=-1
            )
            negative_edges = negative_edges.where(
                negative_node_mask, positive_edges[..., None, :]
            )

        return negative_edges

    def to(self, device: Union[str, torch.device]):
        if self.permuted_probabilities is not None:
            self.permuted_probabilities = self.permuted_probabilities.to(device)
        if self.node_probabilities is not None:
            self.node_probabilities = self.node_probabilities.to(device)
        if self.avoid_edges is not None:
            self.avoid_edges = self.avoid_edges.to(device)
        return self


class RandomNegativeEdges:
    """
    Return randomly sampled (true) negative edges.

    If permutation_option is none, this class uses O(len(avoid_edges)) RAM.
    If permutation_option is head or tail, this class uses O(len(avoid_edges) + num_nodes) RAM.

    TODO: allow for permutation with uniform probability of switching head or tail.
        This can be accomplished using no more than twice the RAM of the current head or tail implementation.

    :param num_nodes: Number of nodes in the graph.
    :param negative_ratio: Number of negatives to sample for each positive.
    :param avoid_edges: If provided, avoid these edges when sampling.
        Should be a LongTensor of shape (..., 2)
    :param device: Specify the device the sampling should be done on.
    :param permutation_option: Specify whether to permute head or tail nodes.
        Default is to perform uniform negative sampling, i.e. PermutationOption.none
    """

    def __init__(
        self,
        num_nodes: int,
        negative_ratio: int,
        avoid_edges: Optional[LongTensor] = None,
        device: Union[None, str, torch.device] = None,
        permutation_option: PermutationOption = PermutationOption.none,
    ):
        self._num_nodes = num_nodes
        self._sample_max = self._num_nodes ** 2
        self.negative_ratio = negative_ratio
        self._permutation_option = PermutationOption(permutation_option)
        if device is not None:
            self._device = device
        else:
            if avoid_edges is None:
                self._device = torch.device("cpu")
            else:
                self._device = avoid_edges.device

        if self.permutation_option != PermutationOption.none:
            self._breakpoints = (
                torch.arange(num_nodes + 1, device=self.device) * num_nodes
            )

        # The following repeats some functionality from pytorch_utils.random.RandomIntsAvoid, might be worth refactoring
        # this to avoid duplication.

        if avoid_edges is None:
            self._buckets = None
        else:
            if (
                self.permutation_option == PermutationOption.head
                or self.permutation_option == PermutationOption.none
            ):
                avoid_ints = convert_edges_to_ints(avoid_edges, self._num_nodes)
            elif self.permutation_option == PermutationOption.tail:
                avoid_ints = convert_edges_to_ints(
                    avoid_edges[..., [1, 0]], self._num_nodes
                )
            else:
                raise ValueError(
                    f"permutation_option={self.permutation_option} not supported."
                )

            avoid_ints = torch.unique(avoid_ints)
            self._sample_max -= len(avoid_ints)  # total number of negatives

            self._buckets = avoid_ints - torch.arange(len(avoid_ints), device=device)
            if self.permutation_option != PermutationOption.none:
                self._breakpoints = self._breakpoints - torch.bucketize(
                    self._breakpoints, avoid_ints, right=False
                )

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2), where [...,0] is the tail
            node index and [...,1] is the head node index.
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """
        negative_int_shape = (*positive_edges.shape[:-1], self.negative_ratio)
        if self.permutation_option == PermutationOption.none:
            sample_idxs = torch.randint(
                self._sample_max, size=negative_int_shape, device=self.device
            )

        else:
            if self.permutation_option == PermutationOption.head:
                fixed_nodes = positive_edges[..., 0]
            elif self.permutation_option == PermutationOption.tail:
                fixed_nodes = positive_edges[..., 1]

            num_neg_per_node = (
                self._breakpoints[fixed_nodes + 1] - self._breakpoints[fixed_nodes]
            )
            sample_idxs = (
                torch.rand(negative_int_shape, device=self.device)
                * num_neg_per_node[..., None]
            ).long() + self._breakpoints[fixed_nodes][..., None]

        if self._buckets is not None:
            sample_idxs += torch.bucketize(sample_idxs, self._buckets, right=True)

        sample_edges = convert_ints_to_edges(sample_idxs, self._num_nodes)
        if self.permutation_option == PermutationOption.tail:
            # In this case, sample_idxs is interpreted as representing an edge (i,j) as j * num_nodes + i.
            # This is the reverse of our normal convention, so we need to swap head and tail.
            sample_edges = sample_edges[..., [1, 0]]
        return sample_edges

    @property
    def device(self):
        return self._device

    @property
    def permutation_option(self):
        return self._permutation_option

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self._random_ints_avoid.to(device)
        self._buckets.to(device)
        self._breakpoints.to(device)
        return self


@attr.s(auto_attribs=True)
class HierarchicalNegativeEdgesBatched:

    edges: Tensor = attr.ib(validator=_validate_edge_tensor)
    weight_strategy: str = "number_of_descendants"  # or "depth"
    negative_ratio: int = 16

    def __attrs_post_init__(self):

        G = nx.DiGraph()

        # assume nodes are numbered contiguously 0 through #nodes, shift by one to add meta root as first node (for now)
        G.add_edges_from((self.edges + 1).tolist())
        self.root_nodes = torch.tensor([node for node, degree in G.in_degree if degree == 0])

        # add edges from meta root (which starts out as index 0, then gets bumped to index 1 by padding which is 0) to
        # root nodes so that everybody has a parent (for indexing purposes only)
        G.add_edges_from([(0, r.item()) for r in self.root_nodes])

        A = nx.adjacency_matrix(G, nodelist=sorted(list(G.nodes)))

        # TODO calculate max depth (max # edges) via dfs
        self.max_depth = 15
        A_ = A.copy()
        for _ in range(self.max_depth):
            A_ += A_ @ A
        A_[A_ > 0] = 1

        # add dummy self-looping row and column at the beginning for padding token and for meta-root
        A = np.vstack([np.zeros((A.shape[0],)), A.todense()])
        A = np.hstack([np.zeros((A.shape[0], 1)), A])
        A[0, 0] = 1

        A_ = np.vstack([np.zeros((A_.shape[0],)), A_.todense()])
        A_ = np.hstack([np.zeros((A_.shape[0], 1)), A_])
        A_[0, 0] = 1

        self.A = csr_matrix(A)
        self.A_ = csr_matrix(A_)
        self.G = G

        self.PAD = 0
        self.METAROOT = 1
        self.base_case_nodes = torch.tensor([self.PAD, self.METAROOT])

        if self.weight_strategy == "number_of_descendants":
            # TODO this is a very inefficient way to collect this info, do it in a single traversal
            # add 1 to # descendants because otherwise leaf nodes will have 0 weight
            node_to_weight = {n - 1: len(nx.descendants(G, n)) + 1 for n in G.nodes if n != 0}
        else:
            # calculate node depths (used as weights); METAROOT has index 0 in G
            # root nodes are at depth 1, successive levels at depths 2, 3, 4...
            node_to_weight = {n - 1: len(p) - 1 for n, p in nx.shortest_path(G, 0).items() if n != 0}

        node_to_weight = torch.FloatTensor([node_to_weight[k] for k in sorted(list(node_to_weight.keys()))]).unsqueeze(-1)
        node_to_weight = torch.cat([node_to_weight, torch.tensor([[0]])], dim=0)
        self.EMB_PAD = node_to_weight.shape[0] - 1
        self.weights = torch.nn.Embedding.from_pretrained(node_to_weight, freeze=True, padding_idx=self.EMB_PAD)

        self.negative_roots = self.precompute_negatives()

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2), where [...,0] is the head
            node index and [...,1] is the tail node index.
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """

        tails = positive_edges[..., 1]
        negative_candidates = self.negative_roots[tails].long()
        negative_candidates_weights = self.weights(negative_candidates).squeeze()
        negative_idxs = torch.tensor(list(WeightedRandomSampler(weights=negative_candidates_weights,
                                                                num_samples=self.negative_ratio,
                                                                replacement=True)))
        negative_nodes = torch.gather(negative_candidates, -1, negative_idxs)
        tails = tails.unsqueeze(-1).expand(-1, self.negative_ratio)
        negative_edges = torch.stack([negative_nodes, tails], dim=-1)
        return negative_edges

    def precompute_negatives(self):

        nodes = torch.arange(2, self.A.shape[0]).unsqueeze(-1)   # exclude pad at index 0 and meta-root at index 1

        negative_roots = self._get_negative_roots(nodes=nodes, negative_roots=torch.zeros((nodes.shape[0], 1)))
        negative_roots = _batch_prune_and_sort(negative_roots, pad=self.PAD) - 2    # shift back pad and meta-root
        negative_roots[negative_roots < 0] = self.EMB_PAD    # prepare for weights lookup

        return negative_roots

    def _get_negative_roots(self, nodes, negative_roots):
        """

        Args:
            nodes: (batch_size, max_nodes), where second dimension corresponds to all parents for the starting node at
                    this level of recursion. Since different starting nodes in a DAG may have different number of parents
                    at a given level of recursion, we pad this dimension with -1 for every starting node dimension with
                    less than max_nodes at the current level of recursion.
            uncles_per_level: dictionary mapping from each level of recursion (going up towards root) to tensor of
                     (batch_size, max_negatives) storing all the roots for negative samples at each level.
            level: parent of positive example is level 0, subtract 1 for each higher level.

        Returns: uncles_per_level

        """

        # base case: checks if every element of nodes is a child of the meta root
        compareview = self.base_case_nodes.repeat(*nodes.shape, 1)
        roots_only = (compareview == nodes.unsqueeze(-1)).sum(-1).all()
        if roots_only:
            return negative_roots

        # to get siblings, we must get all children of the parents, and subtract the parents
        parents = _batch_get_parents_or_children(nodes,
                                                 action="parents",
                                                 adjacency_matrix=self.A)
        children_of_parents = _batch_get_parents_or_children(parents,
                                                             action="children",
                                                             adjacency_matrix=self.A)

        # negatives (siblings) at current level are children of parents that are not current nodes
        #  (i.e. siblings to original nodes if first recursive call, uncles otherwise)
        negative_roots_at_current_level = _batch_set_difference(b1=children_of_parents, b2=nodes)

        # necessary to subtract descendants of current-level roots at each step to only retain highest-level negatives
        descendants_of_negative_roots_at_current_level = _batch_get_parents_or_children(negative_roots_at_current_level,
                                                                                        action="children",
                                                                                        adjacency_matrix=self.A_)

        negative_roots = _batch_set_union(b1=negative_roots, b2=negative_roots_at_current_level)
        negative_roots = _batch_set_difference(b1=negative_roots, b2=descendants_of_negative_roots_at_current_level)

        return self._get_negative_roots(nodes=parents, negative_roots=negative_roots)


def _batch_get_parents_or_children(nodes, adjacency_matrix, action="parents", pad=0, metaroot=1):
    """

    Args:
        nodes: (batch_size, max_length)
        adjacency_matrix: CSR sparse matrix. May be adjacency matrix or full transitive closure (TC) matrix
        action: get "children" (descendants if TC matrix provided) or "parents" (ancestors if TC)
        pad:

    Returns: parents/children for each entry in batch, padded into tensor

    """

    # TODO currently this converts A to a dense matrix - need to figure out advanced indexing in sparse matrix
    if action == "parents":
        # replace METAROOT with PAD if getting parents, because otherwise pack_sequence will break on all-zero parents
        #  of METAROOT which will yield an empty sequence to pack. This won't affect the correctness of algorithm.
        nodes[nodes == metaroot] = pad
        parents = adjacency_matrix.todense()[:, nodes]
        parents, buckets, _ = parents.nonzero()
    elif action == "children":
        # by "parents" here we mean "children" for lack of a general term that encapsulates "parents or children"
        parents = adjacency_matrix.todense()[nodes]
        buckets, _, parents = parents.nonzero()

    parents, buckets = torch.tensor(parents), torch.tensor(buckets)

    ps_bs = torch.vstack([parents, buckets]).T
    sorted_ps_bs = ps_bs[torch.sort(ps_bs[:, -1])[1]]  # sort by bucket
    parents = sorted_ps_bs[:, 0].squeeze()
    bucket_sizes = torch.bincount(sorted_ps_bs[:, 1]).tolist()

    seqs = torch.split(parents, split_size_or_sections=bucket_sizes)
    packed_parents = pack_sequence(seqs, enforce_sorted=False)
    padded_parents, _ = pad_packed_sequence(sequence=packed_parents, batch_first=True, padding_value=pad)

    return padded_parents


def _batch_set_difference(
        b1=torch.tensor([[1, 2, 3, 4, 5],
                         [5, 4, 3, 2, 1]]),
        b2=torch.tensor([[6, 7, 5],
                         [1, 4, 8]])):
    """
    Args:
        b1: (batch_size, n1)
        b2: (batch_size, n2)

    Returns: b1 - b2, i.e. zeroes out all elements in b1 that are found in b2

    """

    compareview = b2.unsqueeze(-2).repeat((1, b1.shape[-1], 1))     # (batch_size, n1, n2)
    b2_mask = (~(compareview == b1.unsqueeze(-1)).any(-1)).long()   # compared with (batch_size, n1, 1)
    diff = b1 * b2_mask                                             # zero out all elements in b1 that are in b2

    return diff


def _batch_set_union(b1, b2):
    """

    Args:
        b1: (batch_size, n1)
        b2: (batch_size, n2)

    Returns: currently just concatenate, don't worry about redundancy

    """

    return torch.cat([b1, b2], dim=-1)


def _batch_prune_and_sort(nodes, pad=0):
    """

    Args:
        nodes: (batch_size, n)

    Returns: moves padding tokens to the end, sorts non-padding indices and removes repeats, e.g.

             [[2,  4,  0,  0,  4,  0],               [[2,  4,  0,  0],
              [7,  0,  7,  1,  2,  3]]      =>        [1,  2,  3,  7]]

        Do this only once after recursion is complete because it's expensive, and not pruning/sorting doesn't affect
        correctness during recursion
    """

    pruned_sorted_rows = [torch.tensor(sorted(list(set(r.tolist()) - {pad}))) for r in nodes]
    pruned_sorted_rows = [r if len(r) > 0 else torch.tensor([pad]) for r in pruned_sorted_rows]
    packed = pack_sequence(pruned_sorted_rows, enforce_sorted=False)
    padded, _ = pad_packed_sequence(sequence=packed, batch_first=True, padding_value=pad)
    return padded


@attr.s(auto_attribs=True)
class HierarchicalNegativeEdgesDebug:

    edges: Tensor = attr.ib(validator=_validate_edge_tensor)

    def __attrs_post_init__(self):

        # create graph with meta-root node
        self.G = nx.DiGraph(self.edges.tolist())
        self.root_nodes = [node for node, degree in self.G.in_degree if degree == 0]
        self.G.add_edges_from([("M", "M")] + [("M", r) for r in self.root_nodes])

        self.precompute_negatives()

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:

        breakpoint()

    def precompute_negatives(self):

        node_to_uncles = dict()
        uncles_per_node = list()
        for node in self.G.nodes:
            if node != "M":
                uncles = self._get_uncles_for_nodes_recursive(nodes={node}, uncles=set())
                node_to_uncles[node] = uncles
                uncles_per_node.append(torch.tensor(uncles))

        packed = pack_sequence(sequences=uncles_per_node, enforce_sorted=False)
        padded, lens = pad_packed_sequence(sequence=packed, batch_first=True, padding_value=-1)

        breakpoint()

    def _get_uncles_for_nodes_recursive(self, nodes, uncles):

        # base case: meta-root
        if len(nodes) == 1 and list(nodes)[0] == "M":
            return sorted(list(uncles))

        parents = self._get_parents(nodes)
        grandparents = self._get_parents(parents)
        children_of_grandparents = self._get_children(grandparents)

        uncles.update(children_of_grandparents - parents - nodes - {"M"})
        uncles -= self._get_descendants(uncles)

        return self._get_uncles_for_nodes_recursive(nodes=parents, uncles=uncles)

    def _get_parents(self, nodes):

        parents = set()
        for node in nodes:
            parents.update(self.G.predecessors(node))

        return parents

    def _get_children(self, nodes):

        children = set()
        for node in nodes:
            children.update({c for c in self.G[node].keys()})

        return children

    def _get_descendants(self, nodes):

        descendants = set()
        for node in nodes:
            if node != "M":
                descendants.update(nx.descendants(self.G, node))

        return descendants


@attr.s(auto_attribs=True)
class GraphDataset(Dataset):
    """
    A map-style dataset, compatible with TensorDataloader.

    :param edges: LongTensor of ids of shape (num,...,2), where edges[...,0] is tail and edges[...,1] is head
        (Typically, shape is (num_edges, 2), but may also be (num_edges, 1 + num_negatives, 2) with positives located in (:, 0, :))
    :param negative_sampling: Callable which takes in a set of (positive) edges and returns negatives.
    """

    edges: Tensor = attr.ib(validator=_validate_edge_tensor)
    num_nodes: int
    negative_sampler: Optional[Callable[[LongTensor,], LongTensor]] = None

    def __attrs_post_init__(self):
        self._device = self.edges.device

    def __getitem__(self, idxs: LongTensor) -> LongTensor:
        """
        :param idxs: LongTensor of shape (...,) indicating the index of the positive edges to select
        :return: LongTensor of shape (..., 1 + num_negatives, 2) where the positives are located in [:,0,:]
        """
        edge_batch = self.edges[idxs]
        if self.negative_sampler is not None:
            negative_edge_batch = self.negative_sampler(edge_batch)
            edge_batch = torch.cat(
                (edge_batch[..., None, :], negative_edge_batch.to(self.device)), dim=-2
            )
        return edge_batch

    def __len__(self):
        return len(self.edges)

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.edges = self.edges.to(device)
        return self


