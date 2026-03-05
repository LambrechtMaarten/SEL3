import functools

import jax
from flax import struct
from jax import numpy as jnp

from src.env import Environment
from src.jax_extra import jarr
from src.numerical_analysis import EulerSolver, DifferentialEquationSolver


# T_i = x_i + r_i * cos(F_i) = output
# r_i'' => approaches R_i
# x_i'' => approaches X_i
# F_i' = o_i + sum(j, w_ij * r_j * sin(F_j - F_i - b_ij))

@struct.dataclass
class CPGState:
    time: float
    frequency: float  # (o)
    outputs: jarr  # (T)
    phases: jarr  # (F)
    amplitudes: jarr  # (r)
    d_amplitudes: jarr  # (r')
    amplitude_goals: jarr  # (R)
    offsets: jarr  # (x)
    d_offsets: jarr  # (x')
    offset_goals: jarr  # (X)
    coupled_phase_biases: jarr  # (b)

    def modulate_random(self, rng=jax.random.PRNGKey(0)):
        ra, rb, rc = jax.random.split(rng, 3)
        amplitude_goals = jax.random.normal(ra, self.amplitude_goals.shape)
        offset_goals = jax.random.normal(rb, self.offset_goals.shape)
        coupled_phase_biases = jax.random.normal(rc, self.coupled_phase_biases.shape)
        frequency = jnp.pi
        return self.modulate(amplitude_goals, offset_goals, coupled_phase_biases, frequency)

    def modulate(self, amplitude_goals: jarr, offset_goals: jarr, coupled_phase_biases: jarr,
                 frequency: float):
        return self.replace(
            amplitude_goals=amplitude_goals, offset_goals=offset_goals, coupled_phase_biases=coupled_phase_biases,
            frequency=frequency
        )


class CPG:
    def __init__(self, adjacency_matrix: jarr, env: Environment,
                 solver: DifferentialEquationSolver = EulerSolver()) -> None:
        self._adjacency_matrix = adjacency_matrix
        self._dt = env.environment_configuration.control_timestep
        self._solver = solver

        self._amplitude_gain = 1
        self._offset_gain = 1

        self.state: CPGState = self.reset()

    def set_state(self, state: CPGState):
        self.state = state

    @property
    def num_oscillators(self) -> int:
        return self._adjacency_matrix.shape[0]

    @staticmethod
    def phase_de(weights: jarr, amplitudes: jarr, phases: jarr, phase_biases: jarr, frequency: float) -> jarr:  # F'
        print(weights.shape, amplitudes.shape, phases.shape, phase_biases.shape)

        @jax.vmap
        def sine_term(phase_i: float | jarr, phase_biases_i: float | jarr) -> jarr:
            return jnp.sin(phases - phase_i - phase_biases_i)

        return frequency + jnp.sum(weights * amplitudes * sine_term(phase_i=phases, phase_biases_i=phase_biases))

    @staticmethod
    def second_order_de(gain: float, modulator: jarr, values: jarr, dot_values: jarr) -> jarr:
        return gain * ((gain / 4) * (modulator - values) - dot_values)

    @staticmethod
    def first_order_de(dot_values: jarr) -> jarr:
        return dot_values

    def reset(self) -> CPGState:
        """
        Creates a new CPG state with everything set to 0
        :return: empty CPG state
        """
        return CPGState(
            time=0.0,
            frequency=0.0,
            outputs=jnp.zeros(self.num_oscillators),
            phases=jnp.zeros(self.num_oscillators),
            amplitudes=jnp.zeros(self.num_oscillators),
            offsets=jnp.zeros(self.num_oscillators),
            d_amplitudes=jnp.zeros(self.num_oscillators),
            d_offsets=jnp.zeros(self.num_oscillators),
            amplitude_goals=jnp.zeros(self.num_oscillators),
            offset_goals=jnp.zeros(self.num_oscillators),
            coupled_phase_biases=jnp.zeros_like(self._adjacency_matrix)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: CPGState = None) -> CPGState:
        if state is None:
            state = self.state

        def step(y, dy):
            return self._solver.solve(current_time=state.time, y=y, delta_time=self._dt, derivative_fn=dy)

        next_phases = step(
            state.phases,
            lambda t, y: self.phase_de(frequency=state.frequency, amplitudes=state.amplitudes, phases=y,
                                       phase_biases=state.coupled_phase_biases, weights=self._adjacency_matrix)
        )
        next_d_amplitudes = step(
            state.d_amplitudes,
            lambda t, y: self.second_order_de(gain=self._amplitude_gain, modulator=state.amplitude_goals,
                                              values=state.amplitudes, dot_values=y)
        )
        next_d_offsets = step(
            state.d_offsets,
            lambda t, y: self.second_order_de(gain=self._offset_gain, modulator=state.offset_goals,
                                              values=state.offsets, dot_values=y)
        )
        next_amplitudes = step(
            state.amplitudes,
            lambda t, y: self.first_order_de(dot_values=state.d_amplitudes)
        )
        next_offsets = step(
            state.offsets,
            lambda t, y: self.first_order_de(dot_values=state.d_offsets)
        )

        next_outputs = next_offsets + next_amplitudes * jnp.cos(next_phases)

        # noinspection PyUnresolvedReferences
        self.state = state.replace(
            phases=next_phases,
            d_amplitudes=next_d_amplitudes,
            amplitudes=next_amplitudes,
            d_offsets=next_d_offsets,
            offsets=next_offsets,
            outputs=next_outputs,
            time=state.time + self._dt
        )
        return self.state

    # todo verwijderen
    # amplitude_goals = jnp.ones(10).at[1::2].set(.2)
    # offset_goals = jnp.zeros(10)
    # coupled_phase_biases = jnp.zeros((10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         if i == j - 1:
    #             coupled_phase_biases = coupled_phase_biases.at[i, j].set(-jnp.pi)
    #         if i == j + 1:
    #             coupled_phase_biases = coupled_phase_biases.at[i, j].set(jnp.pi + i / 10)
    # frequency = jnp.pi
    # cpg_state = cpg_state.modulate(amplitude_goals, offset_goals, coupled_phase_biases, frequency)
