# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import pandas as pd

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction


class Schwefel(SfuFunction):
    """
    The Schwefel Function. See https://www.sfu.ca/~ssurjano/schwef.html for details.
    """

    @property
    def name(self) -> str:
        return 'Schwefel Function'

    def __init__(self, num_dims: int, lb: float = -500, ub: float = 500):


        super(Schwefel, self).__init__(num_dims=num_dims, lb=lb, ub=ub)


        self.x_star = pd.DataFrame([[420.9687 for _ in range(num_dims)]])  # Global minimiser
        if (self.x_star <= self.ub).all().all() and (self.x_star >= self.lb).all().all():
            # self.global_optimum = self.evaluate(self.x_star)[0, 0]  # Global optimum
            self.global_optimum = np.array([0.0002545567494962597])

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
                -500: 0,
                -400: 1,
                -300: 2,
                -200: 3,
                -100: 4,
                0: 5,
                100: 6,
                200: 7,
                300: 8,
                400: 9,
                500: 10
            }

            x_short = np.vectorize(LONG_TO_SHORT.get)(x)
            x_short_shuffled = np.array([flip(_x) for _x in x_short])

            SHORT_TO_LONG = {v: k for k, v in LONG_TO_SHORT.items()}
            x = np.vectorize(SHORT_TO_LONG.get)(x_short_shuffled)

        return (418.9829 * self.num_dims - (x * np.sin(np.sqrt(np.abs(x)))).sum(axis=-1)).reshape(-1, 1)
