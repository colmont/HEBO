import os
import pathlib
from pathlib import Path
from typing import List, Dict, Any

import torch
import pandas as pd
import numpy as np
from numpy.random import RandomState

from mcbo.tasks import TaskBase
from mcbo.tasks.synthetic.maxsat.utils_maxsat import (
    WCNF,
    download_maxsat60_data,
    download_maxsat125_data,
)


class MaxSat(TaskBase):
    """
    A MaxSat benchmark (this class will be subclassed for each MaxSat instance)
    """

    categories = ["OFF", "ON"]

    def __init__(
        self,
        instance_filename: str,
        normalize_weights: bool = True,
        negative_weights: bool = False,
    ):
        """
        Initialize the benchmark function.

        Args:
            instance_filename: the filename of the MaxSat instance
            normalize_weights: whether to normalize the weights to zero mean and unit standard deviation
            negative_weights: whether to use the negative weights (i.e. the weights of the unsatisfied clauses)
        """
        super(MaxSat, self).__init__()

        wcnf = WCNF(
            os.path.join(
                Path(__file__).parent.parent, "data", "maxsat", instance_filename
            )
        )
        self.dim = wcnf.nv

        self.normalize_weights = normalize_weights
        self.negative_weights = negative_weights
        self.weights = np.array(wcnf.weights, dtype=np.float64)
        self.total_weight = self.weights.sum()

        # normalize weights to zero mean and unit standard deviation
        if self.normalize_weights:
            self.weights = (self.weights - self.weights.mean()) / self.weights.std()
        self.clauses = np.zeros((len(wcnf.clauses), self.dim), dtype=np.bool_)

        self.clause_idxs = []
        for i, clause in enumerate(wcnf.clauses):
            clause_idxs = np.abs(np.array(clause)) - 1
            self.clauses[i, clause_idxs] = np.array(clause) > 0
            self.clause_idxs.append(clause_idxs)
        self.random_flip = None

        if self.flip:
            self.random_flip = RandomState(42).choice([False, True], size=self.dim)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        x = x.replace(["OFF", "ON"], [0, 1])
        x = x.to_numpy().squeeze()

        if x.ndim == 1:
            x = x[np.newaxis, :]

        x = x.astype(np.bool_)

        fxs = []
        for _x in x:
            if self.random_flip is not None:
                _x = np.logical_xor(_x, self.random_flip)

            weights_sum = np.sum(
                self.weights
                * [
                    np.any(np.equal(_x[ci], self.clauses[i, ci]))
                    for i, ci in enumerate(self.clause_idxs)
                ]
            )
            if self.negative_weights:
                # weights of unsatisfied clauses
                weight_diff = self.total_weight - weights_sum
                fx = torch.tensor(weight_diff).unsqueeze(-1)
            else:
                fx = -torch.tensor(weights_sum).unsqueeze(-1)
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
                    "categories": MaxSat.categories,
                }
            )
        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(n_stages=self.dim)


class MaxSat60(MaxSat):
    """
    The 60D MaxSat benchmark (see http://www.maxsat.udl.cat/11/benchmarks/index.html for more information)
    """

    @property
    def name(self) -> str:
        return "MaxSat 60"

    def __init__(self, *args, **kwargs):
        if not pathlib.Path("data/maxsat/frb10-6-4.wcnf").exists():
            download_maxsat60_data()
        super().__init__(instance_filename="frb10-6-4.wcnf", *args, **kwargs)


class MaxSat125(MaxSat):
    """
    The 125D MaxSat benchmark (see https://maxsat-evaluations.github.io/2018/benchmarks.html for more information)
    """

    @property
    def name(self) -> str:
        return "MaxSat 125"

    def __init__(self, *args, **kwargs):
        if not pathlib.Path(
            "data/maxsat/cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf"
        ).exists():
            download_maxsat125_data()
        super().__init__(
            instance_filename="cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf",
            *args,
            **kwargs,
        )
