from jax import numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.cpg.cpg_generators.cpg_generators import CPGGenerator
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class FullyConnectedCPGGenerator(CPGGenerator):
    """
    This class has 2 generators per motor
    """

    def generate(self, configuration: Configuration) -> CPG:
        morphology = configuration.simulation.morphology_configuration
        adjacency_matrix = jnp.ones((
            2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0],
            2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0]))

        return CPG(1 * adjacency_matrix, configuration)

    def outputs_to_actions(self, outputs: jarr, configuration: Configuration) -> jarr:
        return outputs

    def modulate_symmetric_rotation(self, cpg_state: CPGState, clockwise_rotations: int) -> CPGState:
        # 30 oscillators = 5 arms × 3 segments × 2 directions → 6 oscillators per arm
        shift = 6 * clockwise_rotations

        def permute_vec(x):
            return jnp.roll(x, shift)

        def permute_mat(x):
            return jnp.roll(jnp.roll(x, shift, axis=0), shift, axis=1)

        return cpg_state.replace(
            amplitude_goals=permute_vec(cpg_state.amplitude_goals),
            offset_goals=permute_vec(cpg_state.offset_goals),
            coupled_phase_biases=permute_mat(cpg_state.coupled_phase_biases),
        )
