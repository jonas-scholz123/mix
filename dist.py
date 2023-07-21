# %%
from abc import ABC, abstractmethod
import numpy as np
from rng import RNGUser


class Continuous(ABC, RNGUser):
    def __init__(self, seed=42) -> None:
        self.seed = seed
        self.reset_rng(seed)

    @abstractmethod
    def sample(self, **kwargs) -> float:
        pass


class UniformContinuous(Continuous):
    def __init__(self, lo, hi, seed=42) -> None:
        super().__init__(seed)
        self.lo = lo
        self.hi = hi

    def sample(self, **kwargs) -> float:
        return self.rng.uniform(self.lo, self.hi, **kwargs)


class PowerContinuous(Continuous):
    def __init__(self, lo=0.0, hi=1.0, power: float = 1.0, seed=42) -> None:
        self.power = power
        self.uniform = UniformContinuous(lo, hi)
        super().__init__(seed)

    def reset_rng(self, seed=42):
        self.uniform.reset_rng(seed)

    def sample(self, **kwargs) -> float:
        return np.power(self.uniform.sample(**kwargs), self.power)
