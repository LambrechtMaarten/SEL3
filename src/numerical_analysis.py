from abc import abstractmethod, ABC
from typing import Callable

from dm_env.specs import Array


class DifferentialEquationSolver(ABC):
    @abstractmethod
    def solve(self,
              current_time: float,
              y: float | Array,
              derivative_fn: Callable[[float, float], float] | Callable[[float, Array], Array],
              delta_time: float
              ) -> float | Array:
        pass


class EulerSolver(DifferentialEquationSolver):
    def solve(self,
              current_time: float,
              y: float,
              derivative_fn: Callable[[float, float], float],
              delta_time: float
              ) -> float:
        slope = derivative_fn(current_time, y)
        next_y = y + delta_time * slope
        return next_y


class RK4Solver(DifferentialEquationSolver):
    def solve(self,
              current_time: float,
              y: float,
              derivative_fn: Callable[[float, float], float],
              delta_time: float
              ) -> float:
        dy1 = derivative_fn(current_time, y)
        dy2 = derivative_fn(current_time + delta_time / 2, y + dy1 * delta_time / 2)
        dy3 = derivative_fn(current_time + delta_time / 2, y + dy2 * delta_time / 2)
        dy4 = derivative_fn(current_time + delta_time, y + dy3 * delta_time)
        return y + (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6 * delta_time
