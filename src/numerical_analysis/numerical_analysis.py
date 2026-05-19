"""Abstract base class for numerical ODE solvers used by the CPG."""

from abc import ABC, abstractmethod
from typing import Callable

from src.jax_extra.jax_extra import jarr


class DifferentialEquationSolver(ABC):
    """Abstract interface for a single-step ODE integrator.

    Concrete subclasses implement specific numerical schemes (Euler, RK4, …).
    The solver is stateless and advances a state variable by one timestep
    given its current value and derivative function.
    """

    @abstractmethod
    def solve(
        self,
        current_time: float,
        y: float | jarr,
        derivative_fn: Callable[[float, float], float] | Callable[[float, jarr], jarr],
        delta_time: float,
    ) -> float | jarr:
        """Advance the state ``y`` by one timestep.

        Args:
            current_time: Current time value passed to the derivative function.
            y: Current value of the state variable (scalar or array).
            derivative_fn: Callable with signature ``(t, y) -> dy/dt``.
            delta_time: Integration step size.

        Returns:
            Estimated value of ``y`` at ``current_time + delta_time``.
        """
