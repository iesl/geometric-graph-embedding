import torch
import pytest
from hypothesis import given, strategies as st
from graph_modeling.models.box import *
from graph_modeling.models.temps import *


@given(
    num_entities=st.integers(100, 200),
    dim=st.integers(20, 100),
    batch_size=st.integers(1, 100),
)
@pytest.mark.parametrize(
    "IntersectionTempClass",
    [ConstTemp, GlobalTemp, PerDimTemp, PerEntityTemp, PerEntityPerDimTemp],
)
@pytest.mark.parametrize(
    "VolumeTempClass",
    [ConstTemp, GlobalTemp, PerDimTemp, PerEntityTemp, PerEntityPerDimTemp],
)
def test_tbox(IntersectionTempClass, VolumeTempClass, num_entities, dim, batch_size):
    """Verify the performant forward pass is accurate using the naive implementation"""
    box_model = TBox(
        num_entities,
        dim,
        intersection_temp=IntersectionTempClass(
            0.01, min=0.0, max=10.0, num_entities=num_entities, dim=dim
        ),
        volume_temp=VolumeTempClass(
            1.0, min=0.0, max=100.0, num_entities=num_entities, dim=dim
        ),
    )
    box_model.train()
    optim = torch.optim.SGD(box_model.parameters(), lr=1e-2)
    idxs = torch.randint(num_entities, size=[batch_size, 2])
    output = box_model(idxs)
    loss = output.sum()
    optim.step()
