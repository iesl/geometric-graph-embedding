# Copyright 2021 The Geometric Graph Embedding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from graph_modeling.training.dataset import *
from graph_modeling.enums import PermutationOption
from pytorch_utils.tensordataloader import TensorDataLoader
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np
import torch
from torch import LongTensor
import os
import pytest
from pathlib import Path
from typing import *

TEST_DIR = Path(os.path.dirname(__file__)).parent  # top-level test directory


class GraphData(NamedTuple):
    path: Optional[Path] = None
    edges: Union[None, LongTensor, np.ndarray] = None
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None


@pytest.fixture()
def random_tree_npz() -> GraphData:
    return GraphData(
        path=TEST_DIR / "data/random_tree.npz", num_nodes=1000, num_edges=999
    )


@pytest.fixture()
def kronecker_graph_npz() -> GraphData:
    return GraphData(
        path=TEST_DIR / "data/kronecker_graph.npz", num_nodes=256, num_edges=4981,
    )


@pytest.fixture(
    params=["random_tree_npz", "kronecker_graph_npz",]
)
def arbitrary_graph(request):
    return request.getfixturevalue(request.param)


def test_edges_and_num_nodes_from_npz(arbitrary_graph: GraphData):
    """Simply test if GraphDatasetUniformNegatives can load from npz without error"""
    edges, num_nodes = edges_and_num_nodes_from_npz(arbitrary_graph.path)
    assert len(edges) == arbitrary_graph.num_edges
    assert num_nodes == arbitrary_graph.num_nodes


def test_get_data_from_graph_dataset(arbitrary_graph):
    g = GraphDataset(*edges_and_num_nodes_from_npz(arbitrary_graph.path))
    t = TensorDataLoader(g, batch_size=100, shuffle=True)
    batch = next(iter(t))
    assert batch.shape == (100, 2)


@pytest.mark.parametrize(
    "permutation_option",
    [PermutationOption.none, PermutationOption.head, PermutationOption.tail],
)
def test_uniform_negatives_dataset(
    arbitrary_graph: GraphData, permutation_option: PermutationOption
):
    positives, num_nodes = edges_and_num_nodes_from_npz(arbitrary_graph.path)
    g = GraphDataset(
        positives,
        num_nodes,
        RandomNegativeEdges(
            num_nodes=num_nodes,
            negative_ratio=100,
            avoid_edges=positives,
            permutation_option=permutation_option,
        ),
    )
    full_graph = torch.zeros((g.num_nodes, g.num_nodes))
    full_graph[positives[:, 0], positives[:, 1]] = 1
    batch = g[torch.arange(len(positives))]
    # Check if [:,0] is where positives are located
    assert (batch[:, 0] == positives).all()
    # Check that all the negatives are zero in the ground-truth graph
    assert (full_graph[batch[:, 1:, 0], batch[:, 1:, 1]] == 0).all()
    # For permuted variants, check that the head or tail is correct
    if permutation_option == PermutationOption.head:
        assert (batch[:, :, 0] == positives[:, None, 0]).all()
    if permutation_option == PermutationOption.tail:
        assert (batch[:, :, 1] == positives[:, None, 1]).all()


@given(
    num_entities=st.integers(100, 2000),
    negative_ratio=st.integers(1, 100),
    batch_size=st.integers(1, 100),
    axes=st.lists(st.integers(1, 10), max_size=2),
)
def test_uniform_random_edges(num_entities, negative_ratio, batch_size, axes):
    random_negatives = RandomEdges(num_entities, negative_ratio)
    random_size = [batch_size, *axes, 2]
    pos_idxs = torch.randint(num_entities, size=tuple(random_size))
    output = random_negatives(pos_idxs)
    assert output.shape == tuple((*pos_idxs.shape[:-1], negative_ratio, 2))


@given(
    num_entities=st.integers(100, 2000),
    negative_ratio=st.integers(1, 100),
    batch_size=st.integers(1, 100),
    axes=st.lists(st.integers(1, 10), max_size=2),
)
def test_permuted_random_edges(num_entities, negative_ratio, batch_size, axes):
    random_negatives = RandomEdges(num_entities, negative_ratio, permuted=True)
    random_size = [batch_size, *axes, 2]
    pos_idxs = torch.randint(num_entities, size=tuple(random_size))
    output = random_negatives(pos_idxs)
    assert output.shape == tuple((*pos_idxs.shape[:-1], negative_ratio, 2))
    pos_idxs = pos_idxs[..., None, :]
    assert (output == pos_idxs).any(dim=-1).all()


@st.composite
def generate_edges(
    draw,
    num_nodes=st.integers(min_value=2, max_value=1_000_000_000),
    num_edges=st.integers(min_value=1, max_value=100_000),
):
    num_nodes = draw(num_nodes)
    edges = draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(draw(num_edges), 2),
            elements=st.integers(min_value=0, max_value=num_nodes - 1),
        )
    )
    return torch.from_numpy(edges), num_nodes


@given(edges_and_num_nodes=generate_edges())
def test_edges_to_ints_and_back(edges_and_num_nodes):
    edges, num_nodes = edges_and_num_nodes
    ints = convert_edges_to_ints(edges, num_nodes)
    assert ints.shape == edges.shape[:-1]
    edges_back = convert_ints_to_edges(ints, num_nodes)
    assert edges_back.shape == edges.shape
    assert (edges == edges_back).all()


@st.composite
def generate_positive_edges(
    draw, num_nodes=st.integers(min_value=2, max_value=1_000_000_000)
):
    """Generates edges with at least one negative still possible"""
    num_nodes = draw(num_nodes)
    max_edges = num_nodes ** 2
    num_edges = draw(st.integers(min_value=1, max_value=min(1_000, max_edges - 1)))
    ints = draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(num_edges,),
            elements=st.integers(min_value=0, max_value=max_edges - 1),
            unique=True,
        )
    )
    return convert_ints_to_edges(torch.from_numpy(ints), num_nodes), num_nodes


@given(
    edges_and_num_nodes=generate_positive_edges(),
    negative_ratio=st.integers(min_value=1, max_value=128),
    batch_size=st.integers(1, 32),
    axes=st.lists(st.integers(1, 16), max_size=2),
)
def test_uniform_random_negative_edges(
    edges_and_num_nodes, negative_ratio, batch_size, axes
):
    avoid_edges, num_nodes = edges_and_num_nodes
    random_edges = RandomNegativeEdges(num_nodes, negative_ratio, avoid_edges)
    # It doesn't matter what the positive edges are, since we are doing uniform negative sampling.
    # We just need any tensor with the given shape.
    positive_edges = torch.empty(size=(batch_size, *axes, 2), dtype=torch.long)
    sample_negative_edges = random_edges(positive_edges)
    negative_shape = (*positive_edges.shape[:-1], negative_ratio, 2)
    assert sample_negative_edges.dtype == torch.long
    assert sample_negative_edges.shape == negative_shape
    sample_edges_ints = convert_edges_to_ints(sample_negative_edges, num_nodes).numpy()
    avoid_edges_ints = convert_edges_to_ints(avoid_edges, num_nodes).numpy()
    assert not np.isin(sample_edges_ints, avoid_edges_ints).any()


@st.composite
def generate_positive_edges_for_permuted_negatives(
    draw,
    num_nodes=st.integers(min_value=2, max_value=1_000),
    permutation_option: PermutationOption = PermutationOption.head,
):
    """
    Generates edges with at least one negative per tail/head still possible (depending on permutation_option).
    Unfortunately, this is a bit circular, as the logic used to generate such edges is similar to that which will
    generate the negatives. One could do this in small cases using rejection sampling, but it is very slow.
    On the other hand, the logic used to generate the valid permutation of edges is basically the same as
    RandomIntsAvoid, which can (and has) already been rigorously tested with more naive sampling approaches.
    """
    permutation_option = PermutationOption(permutation_option)
    num_nodes = draw(num_nodes)
    # first, generate some true negative nodes, one per tail
    true_negatives = (
        draw(
            hnp.arrays(
                dtype=np.int64,
                shape=(num_nodes,),
                elements=st.integers(min_value=0, max_value=num_nodes - 1),
                unique=False,
            )
        )
        + np.arange(num_nodes) * num_nodes
    )

    # After this, there are actually at most num_nodes * (num_nodes - 1) edges remaining
    max_edges = num_nodes * (num_nodes - 1)
    num_edges = draw(st.integers(min_value=1, max_value=min(1_000, max_edges)))
    # we will draw this many edges, and then use the method of RandomIntsAvoid from pytorch_utils.random
    # to generate the actual integers
    ints = draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(num_edges,),
            elements=st.integers(min_value=0, max_value=max_edges - 1),
            unique=True,
        )
    )
    buckets = torch.from_numpy(true_negatives - np.arange(num_nodes))
    ints = torch.from_numpy(ints)
    ints += torch.bucketize(ints, buckets, right=True)

    edges = convert_ints_to_edges(ints, num_nodes)
    if permutation_option == PermutationOption.tail:
        edges = edges[..., [1, 0]]
    return edges, num_nodes


@st.composite
def generate_positive_edges_and_sample(
    draw,
    edges_and_num_nodes=generate_positive_edges(),
    batch_size=st.integers(1, 32),
    axes=st.lists(st.integers(1, 16), max_size=2),
):
    avoid_edges, num_nodes = draw(edges_and_num_nodes)
    # For the purposes of these tests, we will treat all avoid_edges as positives. In practice, avoid_edges and
    # positives may be different sets of edges (eg. if avoid_edges contains diagonal, but positives does not).
    _batch_size = draw(batch_size)
    _axes = draw(axes)
    positive_batch = avoid_edges[
        torch.randint(len(avoid_edges), size=(_batch_size, *_axes,))
    ]
    # Positives would also generally be chosen without replacement, but that also shouldn't matter for these tests.
    return avoid_edges, num_nodes, _batch_size, _axes, positive_batch


@given(
    edges_and_sample=generate_positive_edges_and_sample(
        edges_and_num_nodes=generate_positive_edges_for_permuted_negatives(
            permutation_option=PermutationOption.head
        )
    ),
    negative_ratio=st.integers(min_value=1, max_value=128),
)
def test_permuted_head_random_negative_edges(edges_and_sample, negative_ratio):
    """Test that we can generate random negatives for a given tail node"""
    avoid_edges, num_nodes, batch_size, axes, positive_batch = edges_and_sample
    random_edges = RandomNegativeEdges(
        num_nodes, negative_ratio, avoid_edges, permutation_option="head"
    )
    # We just need any tensor with the given shape.
    sample_negative_edges = random_edges(positive_batch)
    negative_shape = (*positive_batch.shape[:-1], negative_ratio, 2)
    assert sample_negative_edges.dtype == torch.long
    assert sample_negative_edges.shape == negative_shape
    sample_edges_ints = convert_edges_to_ints(sample_negative_edges, num_nodes).numpy()
    avoid_edges_ints = convert_edges_to_ints(avoid_edges, num_nodes).numpy()
    assert not np.isin(sample_edges_ints, avoid_edges_ints).any()
    assert (positive_batch[..., None, 0] == sample_negative_edges[..., 0]).all()


@given(
    edges_and_sample=generate_positive_edges_and_sample(
        edges_and_num_nodes=generate_positive_edges_for_permuted_negatives(
            permutation_option=PermutationOption.tail
        )
    ),
    negative_ratio=st.integers(min_value=1, max_value=128),
)
def test_permuted_tail_random_negative_edges(edges_and_sample, negative_ratio):
    """Test that we can generate random negatives for a given head node"""
    avoid_edges, num_nodes, batch_size, axes, positive_batch = edges_and_sample
    random_edges = RandomNegativeEdges(
        num_nodes, negative_ratio, avoid_edges, permutation_option="tail"
    )
    # We just need any tensor with the given shape.
    sample_negative_edges = random_edges(positive_batch)
    negative_shape = (*positive_batch.shape[:-1], negative_ratio, 2)
    assert sample_negative_edges.dtype == torch.long
    assert sample_negative_edges.shape == negative_shape
    sample_edges_ints = convert_edges_to_ints(sample_negative_edges, num_nodes).numpy()
    avoid_edges_ints = convert_edges_to_ints(avoid_edges, num_nodes).numpy()
    assert not np.isin(sample_edges_ints, avoid_edges_ints).any()
    assert (positive_batch[..., None, 1] == sample_negative_edges[..., 1]).all()
