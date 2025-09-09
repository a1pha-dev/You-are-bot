import random

import numpy as np


class EnvironmentTuner:
    @staticmethod
    def set_all_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)