import abc
from typing import Optional

import torch
from jaxtyping import Float

Array = torch.Tensor

__all__ = ["Flow", "LinearInterpolation"]


class Flow(abc.ABC):
    @abc.abstractmethod
    def derivative(
        self,
        X0: Float[Array, "B d"],
        t: Float[Array, "B"],
        X1: Optional[Float[Array, "B d"]] = None,
    ) -> Float[Array, "B d"]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(
        self,
        X0: Float[Array, "B d"],
        t: Float[Array, "B"],
        X1: Optional[Float[Array, "B d"]] = None,
    ) -> Float[Array, "B d"]:
        raise NotImplementedError()


class LinearInterpolation(Flow):
    def __init__(self, T: float = 1.0) -> None:
        super().__init__()
        self._T = T

    def derivative(
        self,
        X0: Float[Array, "B d"],
        t: Float[Array, "B"],
        X1: Optional[Float[Array, "B d"]] = None,
    ) -> Float[Array, "B d"]:
        if X0.ndim != 2:
            raise ValueError("'X0' should be a 2D array.")
        if X1 is None:
            raise ValueError("'X1' cannot be None.")
        if X1.ndim != 2:
            raise ValueError("'X1' should be a 2D array.")
        if torch.any((t < 0) & (t > self._T)):
            raise ValueError("'t' should be between 0 and T.")

        return X1 - X0

    def __call__(
        self,
        X0: Float[Array, "B d"],
        t: Float[Array, "B"],
        X1: Optional[Float[Array, "B d"]] = None,
    ) -> Float[Array, "B d"]:
        if X0.ndim != 2:
            raise ValueError("'X0' should be a 2D array.")
        if X1 is None:
            raise ValueError("'X1' cannot be None.")
        if X1.ndim != 2:
            raise ValueError("'X1' should be a 2D array.")
        if torch.any((t < 0) & (t > self._T)):
            raise ValueError("'t' should be between 0 and T.")

        return X0 * (self._T - t) + X1 * t
