from typing import Callable

from src.numerical_analysis.numerical_analysis import DifferentialEquationSolver


class RK4Solver(DifferentialEquationSolver):

    def solve(
            self,
            current_time: float,
            y: float,
            derivative_fn: Callable[[float, float], float],
            delta_time: float) -> float:
        dy1 = derivative_fn(current_time, y)
        dy2 = derivative_fn(current_time + delta_time / 2, y + dy1 * delta_time / 2)
        dy3 = derivative_fn(current_time + delta_time / 2, y + dy2 * delta_time / 2)
        dy4 = derivative_fn(current_time + delta_time, y + dy3 * delta_time)
        return y + (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6 * delta_time
