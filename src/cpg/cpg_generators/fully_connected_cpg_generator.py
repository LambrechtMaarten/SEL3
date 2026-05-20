"""Fully connected CPG generator: one oscillator per joint (30 total)."""

from jax import numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.cpg.cpg_generators.cpg_generators import CPGGenerator
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class FullyConnectedCPGGenerator(CPGGenerator):
    """CPG generator with one oscillator per joint degree of freedom.

    Creates a fully connected oscillator network (all-to-all coupling,
    weight = 1) with 2 × arms × segments_per_arm oscillators — one for
    each of the two joint axes per segment.  Oscillator outputs are passed
    directly as joint torques without any remapping.
    """

    def generate(self, configuration: Configuration) -> CPG:
        """Build a fully connected CPG for the morphology in the configuration.

        Args:
            configuration: Global simulation and training configuration.

        Returns:
            A :class:`CPG` with a uniform all-ones adjacency matrix of shape
            ``(2*arms*segments, 2*arms*segments)``.
        """
        morphology = configuration.simulation.morphology_configuration
        adjacency_matrix = jnp.ones(
            (
                2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0],
                2 * morphology.number_of_arms * morphology.number_of_segments_per_arm[0],
            )
        )

        return CPG(1 * adjacency_matrix, configuration)

    def outputs_to_actions(self, outputs: jarr, configuration: Configuration) -> jarr:
        """Return oscillator outputs unchanged as joint actions.

        Args:
            outputs: Oscillator output array, shape ``(N,)``.
            configuration: Unused.

        Returns:
            The same ``outputs`` array passed through unmodified.
        """
        return outputs

    def modulate_symmetric_rotation(
        self, cpg_state: CPGState, clockwise_rotations: int
    ) -> CPGState:
        """Rotate the CPG state by the specified number of arm positions.

        Args:
            cpg_state: CPG state to be rotated.
            clockwise_rotations: Number of arm positions to rotate clockwise.
                Each arm contributes 6 oscillators (3 segments × 2 joints).

        Returns:
            A new :class:`CPGState` with amplitude goals, offset goals, and
            phase biases cyclically shifted by ``6 * clockwise_rotations``.
        """
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
