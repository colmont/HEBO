import torch
import pandas as pd
import numpy as np
from numpy.random import RandomState
from typing import List, Dict, Any

from mcbo.tasks import TaskBase


class Labs(TaskBase):
    """
    The low-autocorrelation binary sequence (LABS) benchmark
    """

    categories = ["OFF", "ON"]

    @property
    def name(self) -> str:
        return "LABS"

    def __init__(self, dim: int = 50, *args, **kwargs):
        """
        Initialize the benchmark function.

        Args:
            dim: the dimensionality of the function
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        super().__init__()

        self.dim = dim
        self.random_flip = None

        if self.flip:
            self.random_flip = RandomState(42).choice([-1, 1], size=dim, replace=True)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        x = x.replace(["OFF", "ON"], [0, 1])
        x = torch.from_numpy(x.to_numpy().squeeze())
        x = x.clone().detach()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2
        fxs = []
        for _x in x:
            # set the 0s to -1
            _x[_x == 0] = -1
            if self.random_flip is not None:
                _x = _x * self.random_flip
            e = 0
            for k in range(1, self.dim):
                e += torch.square(torch.sum(_x[:-k] * _x[k:]))
            fx = -(self.dim**2 / (2 * e)).to(dtype=torch.double).unsqueeze(-1)
            fxs.append(fx)
        fxs = torch.cat(fxs, dim=0).detach().cpu().numpy()
        return fxs[:, None]

    @staticmethod
    def get_static_search_space_params(n_stages: int) -> List[Dict[str, Any]]:
        params = []
        for i in range(1, n_stages + 1):
            params.append(
                {
                    "name": f"stage_{i}",
                    "type": "nominal",
                    "categories": Labs.categories,
                }
            )
        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(n_stages=self.dim)
