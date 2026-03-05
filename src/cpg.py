import functools
from typing import Tuple, List

import jax
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification
from flax import struct
from jax import numpy as jnp

from src.numerical_analysis import EulerSolver, DifferentialEquationSolver


# T_i = x_i + r_i * cos(F_i) = output
# r_i'' => approaches R_i
# x_i'' => approaches X_i
# F_i' = o_i + sum(j, w_ij * r_j * sin(F_j - F_i - b_ij))

@struct.dataclass
class CPGState:
    time: float
    outputs: jnp.ndarray  # (T)
    phases: jnp.ndarray  # (F)
    amplitudes: jnp.ndarray  # (r)
    d_amplitudes: jnp.ndarray  # (r')
    amplitude_goals: jnp.ndarray  # (R)
    offsets: jnp.ndarray  # (x)
    d_offsets: jnp.ndarray  # (x')
    offset_goals: jnp.ndarray  # (X)
    frequencies: jnp.ndarray  # (o)
    coupled_phase_bias: jnp.ndarray  # (b)

    def modulate_cpg(self, leading_arm_index: int, max_joint_limit: float):
        left_rower_arm_indices = [(leading_arm_index - 1) % 5, (leading_arm_index - 2) % 5]
        right_rower_arm_indices = [(leading_arm_index + 1) % 5, (leading_arm_index + 2) % 5]

        R = jnp.zeros_like(self.amplitude_goals)
        X = jnp.zeros_like(self.offset_goals)
        rhos = jnp.zeros_like(self.coupled_phase_bias)
        omegas = jnp.pi * jnp.ones_like(self.frequencies)
        phases_bias_pairs = []

        def get_oscillator_indices_for_arm(arm_index: int) -> Tuple[int, int]:
            return arm_index * 2, arm_index * 2 + 1

        def modulate_leading_arm(_X: jnp.ndarray, _arm_index: int) -> jnp.ndarray:
            ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
            return _X.at[oop_oscillator_index].set(max_joint_limit)

        def modulate_left_rower(_R: jnp.ndarray, _arm_index: int) -> Tuple[jnp.ndarray, List[Tuple[int, int, float]]]:
            ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
            _R = _R.at[ip_oscillator_index].set(max_joint_limit)
            _R = _R.at[oop_oscillator_index].set(max_joint_limit)
            _phase_bias_pairs = [(ip_oscillator_index, oop_oscillator_index, jnp.pi / 2)]
            return _R, _phase_bias_pairs

        def modulate_right_rower(_R: jnp.ndarray, _arm_index: int) -> Tuple[jnp.ndarray, List[Tuple[int, int, float]]]:
            ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
            _R = _R.at[ip_oscillator_index].set(max_joint_limit)
            _R = _R.at[oop_oscillator_index].set(max_joint_limit)
            _phase_bias_pairs = [(ip_oscillator_index, oop_oscillator_index, -jnp.pi / 2)]
            return _R, _phase_bias_pairs

        def phase_biases_second_rowers(_left_arm_index: int, _right_arm_index: int) -> List[Tuple[int, int, float]]:
            left_ip_oscillator_index, _ = get_oscillator_indices_for_arm(arm_index=_left_arm_index)
            right_ip_oscillator_index, _ = get_oscillator_indices_for_arm(arm_index=_right_arm_index)
            _phase_bias_pairs = [(left_ip_oscillator_index, right_ip_oscillator_index, jnp.pi)]
            return _phase_bias_pairs

        X = modulate_leading_arm(_X=X, _arm_index=leading_arm_index)

        R, phb = modulate_left_rower(_R=R, _arm_index=left_rower_arm_indices[0])
        phases_bias_pairs += phb

        R, phb = modulate_left_rower(_R=R, _arm_index=left_rower_arm_indices[1])
        phases_bias_pairs += phb

        R, phb = modulate_right_rower(_R=R, _arm_index=right_rower_arm_indices[0])
        phases_bias_pairs += phb

        R, phb = modulate_right_rower(_R=R, _arm_index=right_rower_arm_indices[1])
        phases_bias_pairs += phb

        phases_bias_pairs += phase_biases_second_rowers(
            _left_arm_index=left_rower_arm_indices[1], _right_arm_index=right_rower_arm_indices[1]
        )

        for oscillator1, oscillator2, bias in phases_bias_pairs:
            rhos = rhos.at[oscillator1, oscillator2].set(bias)
            rhos = rhos.at[oscillator2, oscillator1].set(-bias)

        # noinspection PyUnresolvedReferences
        return self.replace(
            amplitude_goals=R, offset_goals=X, coupled_phase_bias=rhos, frequencies=omegas
        )


class CPG:
    def __init__(self, weights: jnp.ndarray, amplitude_gain: float = 20, offset_gain: float = 20, dt: float = 0.01,
                 solver: DifferentialEquationSolver = EulerSolver()) -> None:
        self._weights = weights
        self._dt = dt

        self._amplitude_gain = amplitude_gain
        self._offset_gain = offset_gain

        self._solver = solver

    @property
    def num_oscillators(self) -> int:
        return self._weights.shape[0]

    @staticmethod
    def phase_de(
            weights: jnp.ndarray,
            amplitudes: jnp.ndarray,
            phases: jnp.ndarray,
            phase_biases: jnp.ndarray,
            omegas: jnp.ndarray
    ) -> jnp.ndarray:
        def sine_term(
                phase_i: float,
                phase_biases_i: float
        ) -> jnp.ndarray:
            return jnp.sin(phases - phase_i - phase_biases_i)

        couplings = jnp.sum(weights * amplitudes * jax.vmap(sine_term)(phase_i=phases, phase_biases_i=phase_biases),
                            axis=1)
        return omegas + couplings

    @staticmethod
    def second_order_de(
            gain: jnp.ndarray,
            modulator: jnp.ndarray,
            values: jnp.ndarray,
            dot_values: jnp.ndarray
    ) -> jnp.ndarray:
        return gain * ((gain / 4) * (modulator - values) - dot_values)

    @staticmethod
    def first_order_de(
            dot_values: jnp.ndarray
    ) -> jnp.ndarray:
        return dot_values

    @staticmethod
    def output(
            offsets: jnp.ndarray,
            amplitudes: jnp.ndarray,
            phases: jnp.ndarray
    ) -> jnp.ndarray:
        return offsets + amplitudes * jnp.cos(phases)

    def reset(self) -> CPGState:
        """
        Creates a new CPG state with everything set to 0
        :return: empty CPG state
        """
        print(self.num_oscillators)
        return CPGState(
            time=0.0,
            outputs=jnp.zeros(self.num_oscillators),
            phases=jnp.zeros(self.num_oscillators),
            amplitudes=jnp.zeros(self.num_oscillators),
            offsets=jnp.zeros(self.num_oscillators),
            d_amplitudes=jnp.zeros(self.num_oscillators),
            d_offsets=jnp.zeros(self.num_oscillators),
            amplitude_goals=jnp.zeros(self.num_oscillators),
            offset_goals=jnp.zeros(self.num_oscillators),
            frequencies=jnp.zeros(self.num_oscillators),
            coupled_phase_bias=jnp.zeros_like(self._weights)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            state: CPGState
    ) -> CPGState:
        # Update phase
        new_phases = self._solver.solve(
            current_time=state.time,
            y=state.phases,
            derivative_fn=lambda
                t,
                y: self.phase_de(
                omegas=state.frequencies,
                amplitudes=state.amplitudes,
                phases=y,
                phase_biases=state.coupled_phase_bias,
                weights=self._weights
            ),
            delta_time=self._dt
        )
        new_dot_amplitudes = self._solver.solve(
            current_time=state.time,
            y=state.d_amplitudes,
            derivative_fn=lambda
                t,
                y: self.second_order_de(
                gain=self._amplitude_gain, modulator=state.amplitude_goals, values=state.amplitudes, dot_values=y
            ),
            delta_time=self._dt
        )
        new_amplitudes = self._solver.solve(
            current_time=state.time,
            y=state.amplitudes,
            derivative_fn=lambda
                t,
                y: self.first_order_de(dot_values=state.d_amplitudes),
            delta_time=self._dt
        )
        new_dot_offsets = self._solver.solve(
            current_time=state.time,
            y=state.d_offsets,
            derivative_fn=lambda
                t,
                y: self.second_order_de(
                gain=self._offset_gain, modulator=state.offset_goals, values=state.offsets, dot_values=y
            ),
            delta_time=self._dt
        )
        new_offsets = self._solver.solve(
            current_time=0,
            y=state.offsets,
            derivative_fn=lambda
                t,
                y: self.first_order_de(dot_values=state.d_offsets),
            delta_time=self._dt
        )

        new_outputs = self.output(offsets=new_offsets, amplitudes=new_amplitudes, phases=new_phases)
        # noinspection PyUnresolvedReferences
        return state.replace(
            phases=new_phases,
            d_amplitudes=new_dot_amplitudes,
            amplitudes=new_amplitudes,
            d_offsets=new_dot_offsets,
            offsets=new_offsets,
            outputs=new_outputs,
            time=state.time + self._dt
        )


def create_cpg(timestep: float) -> CPG:
    ip_oscillator_indices = jnp.arange(0, 10, 2)
    oop_oscillator_indices = jnp.arange(1, 10, 2)

    adjacency_matrix = jnp.zeros((10, 10))
    # Connect oscillators within an arm
    adjacency_matrix = adjacency_matrix.at[ip_oscillator_indices, oop_oscillator_indices].set(1)
    # Connect IP oscillators of neighbouring arms
    adjacency_matrix = adjacency_matrix.at[
        ip_oscillator_indices, jnp.concatenate((ip_oscillator_indices[1:], jnp.array([ip_oscillator_indices[0]])))].set(
        1
    )
    # Connect OOP oscillators of neighbouring arms
    adjacency_matrix = adjacency_matrix.at[oop_oscillator_indices, jnp.concatenate(
        (oop_oscillator_indices[1:], jnp.array([oop_oscillator_indices[0]]))
    )].set(1)

    # Make adjacency matrix symmetric (i.e. make all connections bi-directional)
    adjacency_matrix = jnp.maximum(adjacency_matrix, adjacency_matrix.T)

    return CPG(
        weights=5 * adjacency_matrix,
        amplitude_gain=20,
        offset_gain=20,
        dt=timestep
    )


def map_cpg_outputs_to_actions(
        morphology_configuration: BrittleStarMorphologySpecification,
        cpg_state: CPGState
) -> jnp.ndarray:
    """

    :param cpg_state: state of the cpg
    :return: list of actions
    """
    num_arms = morphology_configuration.number_of_arms
    num_oscillators_per_arm = 2
    num_segments_per_arm = morphology_configuration.number_of_segments_per_arm[0]

    cpg_outputs_per_arm = cpg_state.outputs.reshape((num_arms, num_oscillators_per_arm))
    cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments_per_arm, axis=0)

    actions = cpg_outputs_per_segment.flatten()
    return actions
