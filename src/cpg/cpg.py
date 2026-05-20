"""Central Pattern Generator (CPG) oscillator network for brittle star locomotion."""

import jax
from jax import numpy as jnp

from configs.config import Configuration
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class CPG:
    """Coupled oscillator network implementing a Central Pattern Generator.

    The CPG consists of N oscillators whose phases, amplitudes, and offsets
    evolve according to coupled differential equations.  The output of each
    oscillator drives one joint of the brittle star robot.

    The dynamics follow the standard Matsuoka-style CPG formulation:

    - Phase:     F_i' = o_i + sum_j(w_ij * r_j * sin(F_j - F_i - b_ij))
    - Amplitude: r_i'' approaches R_i via a second-order system
    - Offset:    x_i'' approaches X_i via a second-order system
    - Output:    T_i  = x_i + r_i * cos(F_i)

    where w_ij are coupling weights from the adjacency matrix, b_ij are
    phase biases, R_i / X_i are target amplitude / offset goals, and o_i is
    the intrinsic frequency.

    Attributes:
        _adjacency_matrix: Weight matrix of shape ``(N, N)`` encoding the
            coupling strengths between oscillators.
        _dt: Integration time step (control timestep from configuration).
        _solver: Numerical ODE solver (Euler or RK4).
        _amplitude_gain: Second-order system gain for amplitude convergence.
        _offset_gain: Second-order system gain for offset convergence.
    """

    def __init__(self, adjacency_matrix: jarr, configuration: Configuration):
        """Initialise the CPG with a coupling matrix and configuration.

        Args:
            adjacency_matrix: Square weight matrix of shape ``(N, N)``
                specifying oscillator coupling strengths.
            configuration: Global simulation and training configuration used
                to read the control timestep and ODE solver.
        """
        self._adjacency_matrix = adjacency_matrix
        self._dt = configuration.simulation.environment_configuration.control_timestep
        self._solver = configuration.simulation.solver

        self._amplitude_gain = 20
        self._offset_gain = 20

    @property
    def num_oscillators(self) -> int:
        """Return the number of oscillators in this CPG network.

        Returns:
            Integer number of oscillators (equals the first dimension of the
            adjacency matrix).
        """
        return self._adjacency_matrix.shape[0]

    def reset(self) -> CPGState:
        """Create a new CPG state with all fields initialised to zero.

        Returns:
            A zeroed :class:`CPGState` compatible with this CPG's network size.
        """
        return CPGState.reset(self.num_oscillators, self._adjacency_matrix.shape)

    def step(self, state: CPGState) -> CPGState:
        """Advance the CPG by one control timestep.

        Integrates the phase, amplitude, and offset ODEs using the configured
        numerical solver and computes the new oscillator outputs.

        Args:
            state: Current CPG state.

        Returns:
            Updated CPG state at time ``state.time + dt``.
        """

        def second_order_de(gain, modulator, values, dot_values):
            """Compute the second-order derivative for amplitude/offset convergence.

            Args:
                gain: Convergence gain factor.
                modulator: Target goal value (R_i or X_i).
                values: Current value (r_i or x_i).
                dot_values: Current first derivative.

            Returns:
                Second derivative driving values toward the modulator.
            """
            return gain * ((gain / 4) * (modulator - values) - dot_values)

        def first_order_de(dot_values):
            """Identity first-order derivative (passes through the velocity).

            Args:
                dot_values: Current first derivative to be integrated.

            Returns:
                The derivative unchanged.
            """
            return dot_values

        def phase_de(weights, amplitudes, phases, phase_biases, frequency):
            """Compute phase derivatives for all oscillators.

            Args:
                weights: Coupling weight matrix of shape ``(N, N)``.
                amplitudes: Current oscillator amplitudes, shape ``(N,)``.
                phases: Current oscillator phases, shape ``(N,)``.
                phase_biases: Coupled phase bias matrix, shape ``(N, N)``.
                frequency: Intrinsic frequency scalar.

            Returns:
                Phase derivatives, shape ``(N,)``.
            """
            return frequency + jnp.sum(
                weights
                * amplitudes
                * jax.vmap(lambda fi_i, rho_i: jnp.sin(phases - fi_i - rho_i))(
                    phases, phase_biases
                ),
                axis=1,
            )

        def _step(y, dy):
            """Advance a single state variable by one timestep using the configured solver.

            Args:
                y: Current value of the state variable.
                dy: Derivative function with signature ``(t, y) -> dy``.

            Returns:
                Value of the state variable at the next timestep.
            """
            return self._solver.solve(
                current_time=state.time, y=y, delta_time=self._dt, derivative_fn=dy
            )

        next_phases = _step(
            state.phases,
            lambda t, y: phase_de(
                frequency=state.frequency,
                amplitudes=state.amplitudes,
                phases=y,
                phase_biases=state.coupled_phase_biases,
                weights=self._adjacency_matrix,
            ),
        )
        next_d_amplitudes = _step(
            state.d_amplitudes,
            lambda t, y: second_order_de(
                gain=self._amplitude_gain,
                modulator=state.amplitude_goals,
                values=state.amplitudes,
                dot_values=y,
            ),
        )
        next_d_offsets = _step(
            state.d_offsets,
            lambda t, y: second_order_de(
                gain=self._offset_gain,
                modulator=state.offset_goals,
                values=state.offsets,
                dot_values=y,
            ),
        )
        next_amplitudes = _step(
            state.amplitudes, lambda t, y: first_order_de(dot_values=state.d_amplitudes)
        )
        next_offsets = _step(state.offsets, lambda t, y: first_order_de(dot_values=state.d_offsets))

        next_outputs = next_offsets + next_amplitudes * jnp.cos(next_phases)

        # noinspection PyUnresolvedReferences
        return state.replace(
            phases=next_phases,
            d_amplitudes=next_d_amplitudes,
            amplitudes=next_amplitudes,
            d_offsets=next_d_offsets,
            offsets=next_offsets,
            outputs=next_outputs,
            time=state.time + self._dt,
        )
