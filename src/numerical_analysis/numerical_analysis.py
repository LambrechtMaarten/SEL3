from abc import ABC, abstractmethod
from typing import Callable

from src.jax_extra.jax_extra import jarr


class DifferentialEquationSolver(ABC):
    @abstractmethod
    def solve(
        self,
        current_time: float,
        y: float | jarr,
        derivative_fn: Callable[[float, float], float] | Callable[[float, jarr], jarr],
        delta_time: float,
    ) -> float | jarr:
        pass
