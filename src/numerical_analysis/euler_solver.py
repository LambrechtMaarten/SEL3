from typing import Callable

from src.numerical_analysis.numerical_analysis import DifferentialEquationSolver


class EulerSolver(DifferentialEquationSolver):

    def solve(
            self,
            current_time: float,
            y: float,
            derivative_fn: Callable[[float, float], float],
            delta_time: float) -> float:
        slope = derivative_fn(current_time, y)
        next_y = y + delta_time * slope
        return next_y
