"""Classic fourth-order Runge-Kutta ODE solver."""

from typing import Callable

from src.numerical_analysis.numerical_analysis import DifferentialEquationSolver


class RK4Solver(DifferentialEquationSolver):
    """Classic fourth-order Runge-Kutta (RK4) numerical integrator.

    Provides fourth-order accuracy by evaluating the derivative at four
    intermediate points within the timestep and combining them with weights
    (1/6, 1/3, 1/3, 1/6).
    """

    def solve(
        self,
        current_time: float,
        y: float,
        derivative_fn: Callable[[float, float], float],
        delta_time: float,
    ) -> float:
        """Advance the state by one RK4 step.

        Args:
            current_time: Current time ``t_n``.
            y: Current state value ``y_n``.
            derivative_fn: Callable ``f(t, y)`` returning the derivative.
            delta_time: Integration step size ``dt``.

        Returns:
            Estimated state value ``y_{n+1}`` with fourth-order accuracy.
        """
        dy1 = derivative_fn(current_time, y)
        dy2 = derivative_fn(current_time + delta_time / 2, y + dy1 * delta_time / 2)
        dy3 = derivative_fn(current_time + delta_time / 2, y + dy2 * delta_time / 2)
        dy4 = derivative_fn(current_time + delta_time, y + dy3 * delta_time)
        return y + (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6 * delta_time
