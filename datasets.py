# %%
from typing import Dict, Iterable
from abc import ABC, abstractmethod
from dist import Continuous, PowerContinuous, UniformContinuous

from rng import RNGUser


class Dataset(ABC):
    def __init__(self):
        self.reset()

    def gen_sample(self):
        pass

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def _reset(self):
        pass


class InfiniteDataset(Dataset, RNGUser):
    def __init__(
        self,
        parent_vars: Dict[str, Continuous] = {},
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.parent_vars = parent_vars
        super().__init__()

    @abstractmethod
    def gen_sample(self, **kwargs):
        return

    def sample_parent_vars(self):
        return {name: dist.sample() for name, dist in self.parent_vars.items()}

    def reset(self):
        self._reset()
        self.reset_rng()
        for dist in self.parent_vars.values():
            dist.reset_rng(dist.seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.gen_sample(**self.sample_parent_vars())


class MixedInfiniteDataset(InfiniteDataset):
    def __init__(
        self,
        datasets: Iterable[InfiniteDataset],
        ps: Iterable[float] = None,
        seed: int = 42,
    ) -> None:
        self.datasets = list(datasets)
        self.ps = ps

        if self.ps is not None and len(self.datasets) != len(self.ps):
            raise ValueError("Datasets and probabilities have different lengths.")

        super().__init__({}, seed)

    def gen_sample(self):
        ds = self.rng.choice(self.datasets, p=self.ps)
        return ds.gen_sample(**ds.sample_parent_vars())

    def reset(self):
        for ds in self.datasets:
            ds.reset()
        self.reset_rng()


class FiniteDataset(Dataset):
    def __init__(self) -> None:
        self.idx = 0
        self.size = self.__len__()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __next__(self):
        if self.idx >= self.size:
            raise StopIteration
        item = self.__getitem__(self.idx)
        self.idx += 1
        return item

    def __add__(self, other):
        return MixedFiniteDataset([self, other])

    def reset(self):
        self._reset()
        return


class MixedFiniteDataset(FiniteDataset):
    def __init__(self, datasets: Iterable[FiniteDataset]) -> None:
        self.datasets = list(datasets)
        self.lens = [len(el) for el in self.datasets]
        self.size = sum(self.lens)
        super().__init__()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        for dataset, size in zip(self.datasets, self.lens):
            if idx < size:
                return dataset[idx]
            idx -= size

    def __add__(self, other):
        return MixedFiniteDataset(self.datasets + other.datasets)


class MyFiniteDataset(FiniteDataset):
    def __init__(self) -> None:
        self.data = [0, 1, 2, 3]
        super().__init__()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyInfiniteDataset(InfiniteDataset):
    def __init__(self, lmin, lmax, power) -> None:
        super().__init__(
            {
                "l": UniformContinuous(lmin, lmax),
                "frac": PowerContinuous(power=power),
            }
        )

    def gen_sample(self, l=None, frac=None):
        return frac * l


ds = MyFiniteDataset()
ds2 = MyInfiniteDataset(1, 2, 1)
ds3 = MyFiniteDataset()

ds4 = ds + ds3

ds5 = MyInfiniteDataset(0.1, 0.25, 3)
ds6 = MixedInfiniteDataset([ds2, ds5])

for i, el in enumerate(ds6):
    print(el)
    if i == 5 or i == 10:
        print("reset")
        ds6.reset()
    if i == 20:
        break
