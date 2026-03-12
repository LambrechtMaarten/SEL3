import jax
from jax import numpy as jnp

from configs.config import Configuration
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class CPG:
    def __init__(self, adjacency_matrix: jarr, configuration: Configuration):
        self._adjacency_matrix = adjacency_matrix
        self._dt = configuration.simulation.environment_configuration.control_timestep
        self._solver = configuration.simulation.solver

        self._amplitude_gain = 20
        self._offset_gain = 20

    @property
    def num_oscillators(self) -> int:
        return self._adjacency_matrix.shape[0]

    def reset(self) -> CPGState:
        """
        Creates a new CPG state with everything set to 0
        :return: empty CPG state
        """
        return CPGState.reset(self.num_oscillators, self._adjacency_matrix.shape)

    def step(self, state: CPGState) -> CPGState:
        second_order_de = lambda gain, modulator, values, dot_values: \
            gain * ((gain / 4) * (modulator - values) - dot_values)
        first_order_de = lambda dot_values: \
            dot_values
        phase_de = lambda weights, amplitudes, phases, phase_biases, frequency: \
            frequency + jnp.sum(
                weights *
                amplitudes *
                jax.vmap(lambda fi_i, rho_i: jnp.sin(phases - fi_i - rho_i))(phases, phase_biases),
                axis=1)

        def _step(y, dy):
            return self._solver.solve(current_time=state.time, y=y, delta_time=self._dt, derivative_fn=dy)

        next_phases = _step(
            state.phases,
            lambda t, y: phase_de(frequency=state.frequency, amplitudes=state.amplitudes, phases=y,
                                  phase_biases=state.coupled_phase_biases, weights=self._adjacency_matrix)
        )
        next_d_amplitudes = _step(
            state.d_amplitudes,
            lambda t, y: second_order_de(gain=self._amplitude_gain, modulator=state.amplitude_goals,
                                         values=state.amplitudes, dot_values=y)
        )
        next_d_offsets = _step(
            state.d_offsets,
            lambda t, y: second_order_de(gain=self._offset_gain, modulator=state.offset_goals,
                                         values=state.offsets, dot_values=y)
        )
        next_amplitudes = _step(
            state.amplitudes,
            lambda t, y: first_order_de(dot_values=state.d_amplitudes)
        )
        next_offsets = _step(
            state.offsets,
            lambda t, y: first_order_de(dot_values=state.d_offsets)
        )

        next_outputs = next_offsets + next_amplitudes * jnp.cos(next_phases)

        # noinspection PyUnresolvedReferences
        return state.replace(
            phases=next_phases,
            d_amplitudes=next_d_amplitudes,
            amplitudes=next_amplitudes,
            d_offsets=next_d_offsets,
            offsets=next_offsets,
            outputs=next_outputs,
            time=state.time + self._dt
        )
