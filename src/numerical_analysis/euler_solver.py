"""Explicit first-order Euler ODE solver."""

from typing import Callable

from src.numerical_analysis.numerical_analysis import DifferentialEquationSolver


class EulerSolver(DifferentialEquationSolver):
    """First-order explicit Euler numerical integrator.

    Approximates the next state as ``y_{n+1} = y_n + dt * f(t_n, y_n)``.
    Simple and fast but less accurate than higher-order methods for the same
    step size.
    """

    def solve(
        self,
        current_time: float,
        y: float,
        derivative_fn: Callable[[float, float], float],
        delta_time: float,
    ) -> float:
        """Advance the state by one Euler step.

        Args:
            current_time: Current time ``t_n``.
            y: Current state value ``y_n``.
            derivative_fn: Callable ``f(t, y)`` returning the derivative.
            delta_time: Integration step size ``dt``.

        Returns:
            Estimated state value ``y_{n+1}``.
        """
        slope = derivative_fn(current_time, y)
        next_y = y + delta_time * slope
        return next_y
