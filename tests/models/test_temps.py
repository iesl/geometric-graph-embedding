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

from pytorch_utils import is_broadcastable
from graph_modeling.models.temps import *
import torch
import pytest
from hypothesis import given, strategies as st


@given(
    num_entities=st.integers(100, 200),
    dim=st.integers(20, 100),
    batch_size=st.integers(1, 100),
    axes=st.lists(st.integers(1, 10), max_size=4),
)
@pytest.mark.parametrize(
    "TempClass", [ConstTemp, GlobalTemp, PerDimTemp, PerEntityTemp, PerEntityPerDimTemp]
)
def test_broadcastability(TempClass, num_entities, dim, batch_size, axes):
    temp_module = TempClass(
        init=1.0, min=0.0, max=10.0, num_entities=num_entities, dim=dim
    )
    random_size = [batch_size, *axes]
    idxs = torch.randint(num_entities, size=tuple(random_size))
    output = temp_module(idxs)
    assert is_broadcastable(output.shape, random_size + [2, dim])


@pytest.mark.parametrize("init", [10.0 ** i for i in range(-8, 3)])
@pytest.mark.parametrize("max", [0.1, 1.0, 10.0, 100.0, 1000.0])
def test_bounded_temp(init, max):
    """
    Make sure BoundedTemp initializes the values correctly.
    Note: in general, there will always be settings of (init, min, max) for which the BoundedTemp implementation will
    not be initialized exactly correctly due to numerical issues.
    """
    if init < max:
        bounded_temp = BoundedTemp(1, init, 0.0, max)
        assert torch.allclose(
            torch.tensor(init), bounded_temp(), atol=max * 1e-8, rtol=0
        )
        bounded_temp.temp.data += 1e10
        assert torch.allclose(
            torch.tensor(max), bounded_temp(), atol=max * 1e-8, rtol=0
        )
        bounded_temp.temp.data -= 1e20
        assert torch.allclose(
            torch.tensor(0.0), bounded_temp(), atol=max * 1e-8, rtol=0
        )


def test_bounded_warning(caplog):
    """Make sure BoundedTemp logs a warning if we cannot initialize as requested due to numerical issues"""
    bounded_temp = BoundedTemp(1, 0.0, -0.16589745133188674, 2.2204460492503134e-08)
    assert "BoundedTemp" in caplog.text
