import torch
from torch.nn import Module
from torch import Tensor
from box_training_methods.utils import tiny_value_of_dtype

__all__ = [
    "HardBoxScorer",
]

eps = tiny_value_of_dtype(torch.float)


class HardBoxScorer(Module):

    def __init__(self):
        super().__init__()

    def forward(self, instance_encoding: Tensor, positive_labels_boxes: Tensor, negative_labels_boxes: Tensor):
        """

        Args:
            instance_encoding: box embedding of instance (produced by InstanceEncoder):     (..., 2 [z/Z])
            positive_labels_boxes:                                                          (..., num_pos, 2 [z/Z])
            negative_labels_boxes:                                                          (..., num_neg, 2 [z/Z])

        Returns: positive_scores, negative_scores

        """

        # x : instance
        xz, xZ = instance_encoding[..., 0], instance_encoding[..., 1]       # (..., 1)

        # y : label
        y_pos_z, y_pos_Z = positive_labels_boxes[..., 0], positive_labels_boxes[..., 1]     # (..., num_pos, 1)
        y_neg_z, y_neg_Z = negative_labels_boxes[..., 0], negative_labels_boxes[..., 1]     # (..., num_neg, 1)

        # TODO!!!
        pos_z = None
        pos_Z = None
        neg_z = None
        neg_Z = None

        breakpoint()

        x_log_vol = None

        y_pos_intersection_x_log_vol = None
        y_neg_intersection_x_log_vol = None

        # -log(P(y=1|x)) = -log(V(y&x)/V(x)) = logV(x) - logV(y&x)
        return None
