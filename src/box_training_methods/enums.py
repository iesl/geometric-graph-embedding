from enum import Enum

__all__ = ["PermutationOption"]


class PermutationOption(Enum):
    none = "none"
    head = "head"
    tail = "tail"
