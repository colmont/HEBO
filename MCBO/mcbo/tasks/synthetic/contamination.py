import torch
import pandas as pd
import numpy as np
from numpy.random import RandomState
from typing import List, Dict, Any

from mcbo.tasks import TaskBase


class Contamination(TaskBase):
    """
    Contamination Control Problem with the simplest graph
    """

    categories = ["OFF", "ON"]

    @property
    def name(self) -> str:
        return "Contamination"

    def __init__(self, effective_dim: int = 25, ambient_dim: int = 25, *args, **kwargs):
        """
        Initialize the benchmark function.

        Args:
            effective_dim: the effective dimensionality of the function
            ambient_dim: the ambient dimensionality of the function
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        super().__init__()

        self.ambient_dim = ambient_dim
        self.effective_dim = effective_dim
        self.lamda = 1e-2
        self.n_vertices = np.array([2] * self.effective_dim)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat(
            [
                self.suggested_init,
                sample_init_points(
                    self.n_vertices, 20 - self.suggested_init.size(0), random_seed=42
                ),
            ],
            dim=0,
        )
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []

        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(
                torch.ones(n_v - 1), 1
            )
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.linalg.eigh(laplacian)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(
            random_seed=42
        )

        if self.flip:
            self.random_flip = torch.tensor(
                RandomState(42).choice([True, False], size=(self.effective_dim,))
            )
        else:
            self.random_flip = None

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        x = x.replace(["OFF", "ON"], [0, 1])
        x = torch.from_numpy(x.to_numpy().squeeze())
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == self.ambient_dim
        fxs = (
            torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0)
            .to(dtype=torch.double)
            .detach()
            .cpu()
            .numpy()
        )
        return fxs[:, None]

    def _evaluate_single(self, x):
        assert x.dim() == 1
        assert x.numel() == self.ambient_dim
        if x.dim() == 2:
            x = x.squeeze(0)
        x = x[: self.effective_dim]
        if self.random_flip is not None:
            x = torch.logical_xor(x, self.random_flip).to(dtype=torch.float)
        evaluation = _contamination(
            x=(x.cpu() if x.is_cuda else x).numpy(),
            cost=np.ones(x.numel()),
            init_Z=self.init_Z,
            lambdas=self.lambdas,
            gammas=self.gammas,
            U=0.1,
            epsilon=0.05,
        )
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()

    @staticmethod
    def get_static_search_space_params(n_stages: int) -> List[Dict[str, Any]]:
        params = []
        for i in range(1, n_stages + 1):
            params.append(
                {
                    "name": f"stage_{i}",
                    "type": "nominal",
                    "categories": Contamination.categories,
                }
            )
        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(n_stages=self.ambient_dim)


def generate_contamination_dynamics(
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the contamination dynamics for the Contamination benchmark.

    Args:
        random_seed: the random seed to use for the random number generator

    Returns:
        the initial contamination, the contamination lambdas, and the restoration gammas

    """
    n_stages = 25
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_Z = np.random.RandomState(random_seed).beta(
        init_alpha, init_beta, size=(n_simulations,)
    )
    lambdas = np.random.RandomState(random_seed).beta(
        contam_alpha, contam_beta, size=(n_stages, n_simulations)
    )
    gammas = np.random.RandomState(random_seed).beta(
        restore_alpha, restore_beta, size=(n_stages, n_simulations)
    )

    return init_Z, lambdas, gammas


def sample_init_points(n_vertices: int, n_points: int, random_seed: int = None):
    """
    Sample initial points for the Contamination benchmark.

    Args:
        n_vertices:
        n_points:
        random_seed:

    Returns:
        torch.Tensor: the initial points

    """
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    init_points = torch.empty(0).long()
    for _ in range(n_points):
        init_points = torch.cat(
            [
                init_points,
                torch.cat(
                    [torch.randint(0, int(elm), (1, 1)) for elm in n_vertices], dim=1
                ),
            ],
            dim=0,
        )
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return init_points


def _contamination(x: np.ndarray, cost, init_Z, lambdas, gammas, U, epsilon):
    assert x.size == 25

    rho = 1.0
    n_simulations = 100

    Z = np.zeros((x.size, n_simulations))
    Z[0] = (
        lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
    )
    for i in range(1, 25):
        Z[i] = (
            lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1])
            + (1.0 - gammas[i] * x[i]) * Z[i - 1]
        )

    below_threshold = Z < U
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)
