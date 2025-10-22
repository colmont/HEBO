# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class Sphere(SfuFunction):
    """
    The Sphere Function. See https://www.sfu.ca/~ssurjano/spheref.html for details.
    """

    @property
    def name(self) -> str:
        return 'Sphere Function'

    def __init__(self, num_dims: int, lb: float = -5.12, ub: float = 5.12):


        super(Sphere, self).__init__(num_dims=num_dims, lb=lb, ub=ub)


        self.x_star = pd.DataFrame([[0 for _ in range(num_dims)]])  # Global minimiser
        if (self.x_star <= self.ub).all().all() and (self.x_star >= self.lb).all().all():
            # self.global_optimum = self.evaluate(self.x_star)[0, 0]  # Global optimum
            self.global_optimum = np.array([0.0])

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.num_dims
        assert (x <= self.ub).all().all()
        assert (x >= self.lb).all().all()

        x = x.to_numpy().astype(float)

        if self.flip:
            def flip(x):
                assert x.ndim == 1
                n_choice = 11
                n_stages = 20
                x = (x + np.random.RandomState(42).choice(n_choice, n_stages)) % n_choice
                return x

            LONG_TO_SHORT = {
                -5.12: 0,
                -4.096: 1,
                -3.072: 2,
                -2.048: 3,
                -1.024: 4,
                0.0: 5,
                1.024: 6,
                2.048: 7,
                3.072: 8,
                4.096: 9,
                5.12: 10
            }

            x = np.round(x, decimals=4)
            x_short = np.vectorize(LONG_TO_SHORT.get)(x)
            x_short_shuffled = np.array([flip(_x) for _x in x_short])

            SHORT_TO_LONG = {v: k for k, v in LONG_TO_SHORT.items()}
            x = np.vectorize(SHORT_TO_LONG.get)(x_short_shuffled)

        return (x ** 2).sum(axis=-1).reshape(-1, 1)
