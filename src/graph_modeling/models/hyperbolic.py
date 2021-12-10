import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch.nn import Module

from .temps import BoundedTemp

__all__ = [
    "Lorentzian",
    "LorentzianDistance",
    "LorentzianScore",
    "squared_lorentzian_distance",
    "lorentzian_inner_product",
    "hyperboloid_vector",
    "HyperbolicEntailmentCones",
]


class Lorentzian(Module):
    """
    This embedding model uses the (symmetric) squared lorentzian distance function from Law et al. 2019
    (http://proceedings.mlr.press/v97/law19a/law19a.pdf) for training, and an adaptation of the score function from
    equation (8) of Nickel & Kiela 2017 (https://arxiv.org/pdf/1705.08039.pdf) for evaluation. Namely, our
    evaluation function will be

        -(1 + alpha (||u||^2 - ||v||^2)) ||u - v||_L^2

    where ||.||_L is the Lorentzian distance.

    :param alpha: penalty for distance, where higher alpha emphasises distance as a determining factor
        for edge direction more
    :param beta: -1/curvature
    """

    def __init__(
        self, num_entity: int, dim: int, alpha: float = 5.0, beta: float = 1.0
    ):
        super().__init__()
        self.embeddings_in = torch.nn.Embedding(num_entity, dim)
        self.alpha = alpha
        self.beta = beta

    def forward(self, idxs: LongTensor) -> Tensor:
        """
        Returns the score of edges between the nodes in `idxs`.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :return: score
        """
        euclidean_embeddings = self.embeddings_in(idxs)
        hyperboloid_embeddings = hyperboloid_vector(euclidean_embeddings, self.beta)
        dist = squared_lorentzian_distance(
            hyperboloid_embeddings[..., 0, :],
            hyperboloid_embeddings[..., 1, :],
            self.beta,
        )
        if self.training:
            return dist + 1e-5  # flip sign for distance loss
        else:
            euclidean_norm = euclidean_embeddings.pow(2).sum(dim=-1)
            return (
                -(1 + self.alpha * (euclidean_norm[..., 0] - euclidean_norm[..., 1]))
                * dist
            )


class LorentzianDistance(Module):
    """
    This embedding returns a score <=0 for a given edge, where higher is better.
    The score is given by

        -||u - v||_L^2 - alpha softplus(||u|| - ||v||)

    where ||.||_L is the Lorentzian distance.

    :param alpha: penalty for distance, where higher alpha emphasises distance as a determining factor
        for edge direction more
    :param beta: -1/curvature
    """

    def __init__(
        self, num_entity: int, dim: int, alpha: float = 5.0, beta: float = 1.0
    ):
        super().__init__()
        self.embeddings_in = torch.nn.Embedding(num_entity, dim)
        self.alpha = alpha
        self.beta = beta

    def forward(self, idxs: LongTensor) -> Tensor:
        """
        Returns the score the edges between the nodes in `idxs`.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :return: score
        """
        euclidean_embeddings = self.embeddings_in(idxs)
        hyperboloid_embeddings = hyperboloid_vector(euclidean_embeddings, self.beta)
        dist = squared_lorentzian_distance(
            hyperboloid_embeddings[..., 0, :],
            hyperboloid_embeddings[..., 1, :],
            self.beta,
        )
        euclidean_norm = torch.norm(euclidean_embeddings, dim=-1)
        return -dist - self.alpha * F.softplus(
            euclidean_norm[..., 0] - euclidean_norm[..., 1]
        )


class LorentzianScore(Module):
    """
    This embedding model combines the score function from equation (8) of Nickel & Kiela 2017
    (https://arxiv.org/pdf/1705.08039.pdf) with the Lorentzian distance from Law et al. 2019
    (http://proceedings.mlr.press/v97/law19a/law19a.pdf). The score function is:

        -(1 + alpha (||u||^2 - ||v||^2)) ||u - v||_L^2

    where ||.||_L is the Lorentzian distance.

    :param alpha: penalty for distance, where higher alpha emphasises distance as a determining factor
        for edge direction more
    :param beta: -1/curvature
    """

    def __init__(
        self, num_entity: int, dim: int, alpha: float = 1e-3, beta: float = 1.0
    ):
        super().__init__()
        self.embeddings_in = torch.nn.Embedding(num_entity, dim)
        self.alpha = alpha
        self.beta = beta

    def forward(self, idxs: LongTensor) -> Tensor:
        """
        Returns the score the edges between the nodes in `idxs`.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :return: score
        """
        euclidean_embeddings = self.embeddings_in(idxs)
        hyperboloid_embeddings = hyperboloid_vector(euclidean_embeddings, self.beta)
        dist = squared_lorentzian_distance(
            hyperboloid_embeddings[..., 0, :],
            hyperboloid_embeddings[..., 1, :],
            self.beta,
        )
        euclidean_norm = euclidean_embeddings.pow(2).sum(dim=-1)
        return (
            -(1 + self.alpha * (euclidean_norm[..., 0] - euclidean_norm[..., 1])) * dist
        )


def squared_lorentzian_distance(a: Tensor, b: Tensor, beta: float = 1.0) -> Tensor:
    """
    Given vectors a, b in H^{d, beta} we calculate the squared Lorentzian distance:
        ||a - b||_L = -2 beta - 2 <a, b>_L
    where <a, b>_L is the Lorentzian inner-product.

    :param a: tensor of shape (..., d+1) representing vectors in H^{d, beta}
    :param b: tensor of shape (..., d+1) representing vectors in H^{d, beta}
    :param beta: -1/curvature
    :return: tensor of shape (...,) representing the distance ||a - b||_L
    """
    # First we map from RR^d to H^(d, beta) by calculating the first coordinate
    # Note: if we're using this for gradient descent, we can probably remove the radius from this
    # subsequent calculation. We could also simply perform the inner product, and use this as a direct
    # replacement of the inner product on RR^d...
    return -2 * beta - 2 * lorentzian_inner_product(a, b)


def lorentzian_inner_product(a: Tensor, b: Tensor) -> Tensor:
    """
    Given vectors a, b in H^{d, beta} we calculate the Lorentzian inner product:
        -a_0 b_0 + sum_i a_i b_i

    :param a: tensor of shape (..., d+1) representing vectors in H^{d, beta}
    :param b: tensor of shape (..., d+1) representing vectors in H^{d, beta}
    :return: tensor of shape (...,) representing the inner product <a, b>_L
    """
    prod = a * b
    prod[..., 0] *= -1
    return torch.sum(prod, dim=-1)


def hyperboloid_vector(u: Tensor, beta: float = 1.0) -> Tensor:
    """
    Given a vector u in RR^d, we map it to a vector a in the hyperboloid H^{d, beta} (where -1/beta is the
    curvature) by setting
        a_0 = sqrt(||a||^2 + radius),   a_i = u_i

    :param u: tensor of shape (..., d) representing vectors in RR^d
    :param beta: -1/curvature
    :return: tensor of shape (...,d+1) representing a vector in H^{d, beta}
    """
    a_0 = (u.pow(2).sum(dim=-1, keepdim=True) + beta).sqrt()
    return torch.cat((a_0, u), dim=-1)


class HyperbolicEntailmentCones(Module):
    """
    This embedding model represents entities as cones in the Poincare disk, where the aperture of the cone is dependent
    on the origin in such a way as to preserve transitivity with respect to containment.
    (See https://arxiv.org/pdf/1804.01882.pdf)

    :param relative_cone_aperture_scale: Number in (0,1] which is the relative size of the aperture with respect to
        distance from origin. Our implementation is such that
            K = relative_cone_aperture_scale * eps_bound / (1 - eps_bound^2)
        (see eq. (25) in the above paper as to why this is required)
    :param eps_bound: bounds vectors in the annulus between eps and 1-eps.
    """

    def __init__(
        self,
        num_entity: int,
        dim: int,
        relative_cone_aperture_scale: float = 1.0,
        eps_bound: float = 0.1,
    ):
        super().__init__()
        self.eps_bound = eps_bound
        assert 0 < self.eps_bound < 0.5
        self.cone_aperature_scale = (
            relative_cone_aperture_scale * self.eps_bound / (1 - self.eps_bound ** 2)
        )

        self.angles = torch.nn.Embedding(num_entity, dim)
        initial_radius_range = 0.9 * (1 - 2 * self.eps_bound)
        initial_radius = 0.5 + initial_radius_range * (torch.rand(num_entity) - 0.5)
        self.radii = BoundedTemp(
            num_entity, initial_radius, self.eps_bound, 1 - self.eps_bound,
        )

    def forward(self, idxs: LongTensor) -> Tensor:
        """
        Returns the score of edges between the nodes in `idxs`.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :return: score
        """
        angles = F.normalize(self.angles(idxs), p=2, dim=-1)
        radii = self.radii(idxs)
        vectors = radii[..., None] * angles

        # test_vectors_radii = torch.linalg.norm(vectors, dim=-1)
        # assert (test_vectors_radii > self.eps_bound).all()
        # assert (test_vectors_radii < 1 - self.eps_bound).all()
        # assert torch.isclose(test_vectors_radii, radii).all()

        radii_squared = radii ** 2
        euclidean_dot_products = (vectors[..., 0, :] * vectors[..., 1, :]).sum(dim=-1)
        euclidean_distances = torch.linalg.norm(
            vectors[..., 0, :] - vectors[..., 1, :], dim=-1
        )

        parent_aperature_angle_sin = (
            self.cone_aperature_scale * (1 - radii_squared[..., 0]) / radii[..., 0]
        )
        # assert (parent_aperature_angle_sin >= -1).all()
        # assert (parent_aperature_angle_sin <= 1).all()
        parent_aperature_angle = torch.arcsin(parent_aperature_angle_sin)

        min_angle_parent_rotation_cos = (
            euclidean_dot_products * (1 + radii_squared[..., 0])
            - radii_squared[..., 0] * (1 + radii_squared[..., 1])
        ) / (
            radii[..., 0]
            * euclidean_distances
            * torch.sqrt(
                1
                + radii_squared[..., 0] * radii_squared[..., 1]
                - 2 * euclidean_dot_products
            )
            + 1e-22
        )
        # assert (min_angle_parent_rotation_cos >= -1).all()
        # assert (min_angle_parent_rotation_cos <= 1).all()
        # original implementation clamps this value from -1+eps to 1-eps, however it seems as though [-1, 1] is all that
        # is required.
        min_angle_parent_rotation = torch.arccos(
            min_angle_parent_rotation_cos.clamp(-1, 1)
        )
        # The energy in the original formulation is clamped, which means gradients for negative examples may be squashed.
        return (parent_aperature_angle - min_angle_parent_rotation).clamp_max(0)
        # Two potential alternatives:
        # return -F.softplus(-parent_aperature_angle + min_angle_parent_rotation, beta=40)
        # (beta would have to be tuned)
        # The following is recommended in https://arxiv.org/pdf/1902.04335.pdf, however this requires a corresponding
        # adjustment to the loss function.
        # return parent_aperature_angle - min_angle_parent_rotation
