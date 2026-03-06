from abc import ABC, abstractmethod
from typing import Tuple, List

import jax.numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG, CPGState
from src.jax_extra import jarr
from src.numerical_analysis import RK4Solver, EulerSolver


class CPGGenerator(ABC):
    """
    This class contains the functions to generate a CPG and to then transform its output into actions, so an
    instance determines which generators map to which motors
    """

    def __init__(self):
        self.configuration: Configuration | None = None

    def set_configuration(self, configuration: Configuration):
        self.configuration = configuration

    @abstractmethod
    def generate(self) -> CPG:
        pass

    @abstractmethod
    def outputs_to_actions(self, outputs: jarr) -> jarr:
        pass


class BasicCPGGenerator(CPGGenerator):
    """
    This class has generators per arm, one "out of plane" and one "in plane" (stolen from tutorial)
    """

    def generate(self) -> CPG:
        ip_oscillator_indices = jnp.arange(0, 10, 2)
        oop_oscillator_indices = jnp.arange(1, 10, 2)

        adjacency_matrix = jnp.zeros((10, 10))
        # Connect oscillators within an arm
        adjacency_matrix = adjacency_matrix.at[ip_oscillator_indices, oop_oscillator_indices].set(1)
        # Connect IP oscillators of neighbouring arms
        adjacency_matrix = adjacency_matrix.at[
            ip_oscillator_indices, jnp.concatenate(
                (ip_oscillator_indices[1:], jnp.array([ip_oscillator_indices[0]])))].set(
            1
        )
        # Connect OOP oscillators of neighbouring arms
        adjacency_matrix = adjacency_matrix.at[oop_oscillator_indices, jnp.concatenate(
            (oop_oscillator_indices[1:], jnp.array([oop_oscillator_indices[0]]))
        )].set(1)

        # Make adjacency matrix symmetric (i.e. make all connections bi-directional)
        adjacency_matrix = jnp.maximum(adjacency_matrix, adjacency_matrix.T)

        return CPG(5 * adjacency_matrix, self.configuration, EulerSolver())

    def outputs_to_actions(self, outputs: jarr) -> jarr:
        num_arms = self.configuration.simulation.morphology_configuration.number_of_arms
        num_oscillators_per_arm = 2
        num_segments_per_arm = self.configuration.simulation.morphology_configuration.number_of_segments_per_arm[0]

        cpg_outputs_per_arm = outputs.reshape((num_arms, num_oscillators_per_arm))
        cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments_per_arm, axis=0)

        actions = cpg_outputs_per_segment.flatten()
        return actions

    @staticmethod
    def modulate_cpg(
            cpg_state: CPGState,
            leading_arm_index: int,
            max_joint_limit: float) -> CPGState:
        def get_oscillator_indices_for_arm(
                arm_index: int
        ) -> Tuple[int, int]:
            return arm_index * 2, arm_index * 2 + 1

        left_rower_arm_indices = [(leading_arm_index - 1) % 5, (leading_arm_index - 2) % 5]
        right_rower_arm_indices = [(leading_arm_index + 1) % 5, (leading_arm_index + 2) % 5]

        amplitude_goals = jnp.zeros_like(cpg_state.amplitude_goals)
        offset_goals = jnp.zeros_like(cpg_state.offsets)
        coupled_phase_biases = jnp.zeros_like(cpg_state.coupled_phase_biases)
        phases_bias_pairs = []

        def modulate_leading_arm(
                _X: jnp.ndarray,
                _arm_index: int
        ) -> jnp.ndarray:
            ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
            return _X.at[oop_oscillator_index].set(max_joint_limit)

        def modulate_left_rower(
                _R: jnp.ndarray,
                _arm_index: int
        ) -> Tuple[jnp.ndarray, List[Tuple[int, int, float]]]:
            ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
            _R = _R.at[ip_oscillator_index].set(max_joint_limit)
            _R = _R.at[oop_oscillator_index].set(max_joint_limit)
            _phase_bias_pairs = [(ip_oscillator_index, oop_oscillator_index, jnp.pi / 2)]
            return _R, _phase_bias_pairs

        def modulate_right_rower(
                _R: jnp.ndarray,
                _arm_index: int
        ) -> Tuple[jnp.ndarray, List[Tuple[int, int, float]]]:
            ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
            _R = _R.at[ip_oscillator_index].set(max_joint_limit)
            _R = _R.at[oop_oscillator_index].set(max_joint_limit)
            _phase_bias_pairs = [(ip_oscillator_index, oop_oscillator_index, -jnp.pi / 2)]
            return _R, _phase_bias_pairs

        def phase_biases_second_rowers(
                _left_arm_index: int,
                _right_arm_index: int
        ) -> List[Tuple[int, int, float]]:
            left_ip_oscillator_index, _ = get_oscillator_indices_for_arm(arm_index=_left_arm_index)
            right_ip_oscillator_index, _ = get_oscillator_indices_for_arm(arm_index=_right_arm_index)
            _phase_bias_pairs = [(left_ip_oscillator_index, right_ip_oscillator_index, jnp.pi)]
            return _phase_bias_pairs

        offset_goals = modulate_leading_arm(_X=offset_goals, _arm_index=leading_arm_index)

        amplitude_goals, phb = modulate_left_rower(_R=amplitude_goals, _arm_index=left_rower_arm_indices[0])
        phases_bias_pairs += phb

        amplitude_goals, phb = modulate_left_rower(_R=amplitude_goals, _arm_index=left_rower_arm_indices[1])
        phases_bias_pairs += phb

        amplitude_goals, phb = modulate_right_rower(_R=amplitude_goals, _arm_index=right_rower_arm_indices[0])
        phases_bias_pairs += phb

        amplitude_goals, phb = modulate_right_rower(_R=amplitude_goals, _arm_index=right_rower_arm_indices[1])
        phases_bias_pairs += phb

        phases_bias_pairs += phase_biases_second_rowers(
            _left_arm_index=left_rower_arm_indices[1], _right_arm_index=right_rower_arm_indices[1]
        )

        for oscillator1, oscillator2, bias in phases_bias_pairs:
            coupled_phase_biases = coupled_phase_biases.at[oscillator1, oscillator2].set(bias)
            coupled_phase_biases = coupled_phase_biases.at[oscillator2, oscillator1].set(-bias)

        # noinspection PyUnresolvedReferences
        return cpg_state.replace(
            amplitude_goals=amplitude_goals, offset_goals=offset_goals, coupled_phase_biases=coupled_phase_biases,
            frequency=jnp.pi,
        )


class FullyConnectedCPGGenerator(CPGGenerator):
    """
    This class has 2 generators per motor
    """

    def generate(self) -> CPG:
        morphology = self.configuration.simulation.morphology_configuration
        adjacency_matrix = jnp.ones((
            2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0],
            2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0]))

        return CPG(1 * adjacency_matrix, self.configuration)

    def outputs_to_actions(self, outputs: jarr) -> jarr:
        return outputs
