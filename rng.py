import numpy as np


class RNGUser:
    def reset_rng(self, seed=42):
        self.rng = np.random.default_rng(seed)
