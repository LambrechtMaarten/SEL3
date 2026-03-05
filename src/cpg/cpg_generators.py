from abc import ABC, abstractmethod

import jax.numpy as jnp
from src.cpg.cpg import CPG, CPGState
from src.env import Environment
from src.jax_extra import jarr


class CPGGenerator(ABC):
    """
    This class contains the functions to generate a CPG and to then transform its output into actions, so an
    instance determines which generators map to which motors
    """

    def __init__(self, env: Environment):
        self.env = env

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

        return CPG(1 * adjacency_matrix, self.env)

    def outputs_to_actions(self, outputs: jarr) -> jarr:
        num_arms = self.env.morphology_configuration.number_of_arms
        num_oscillators_per_arm = 2
        num_segments_per_arm = self.env.morphology_configuration.number_of_segments_per_arm[0]

        cpg_outputs_per_arm = outputs.reshape((num_arms, num_oscillators_per_arm))
        cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments_per_arm, axis=0)

        actions = cpg_outputs_per_segment.flatten()
        return actions

class FullyConnecyedCPGGenerator(CPGGenerator):
    """
    This class has 2 generators per motor
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

        return CPG(1 * adjacency_matrix, self.env)

    def outputs_to_actions(self, outputs: jarr) -> jarr:
        num_arms = self.env.morphology_configuration.number_of_arms
        num_oscillators_per_arm = 2
        num_segments_per_arm = self.env.morphology_configuration.number_of_segments_per_arm[0]

        cpg_outputs_per_arm = outputs.reshape((num_arms, num_oscillators_per_arm))
        cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments_per_arm, axis=0)

        actions = cpg_outputs_per_segment.flatten()
        return actions
