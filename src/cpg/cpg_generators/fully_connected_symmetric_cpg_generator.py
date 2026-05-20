"""Fully connected symmetric CPG generator exploiting the robot's bilateral symmetry."""

from jax import numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.cpg.cpg_generators.cpg_generators import CPGGenerator
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class FullyConnectedSymmetricCPGGenerator(CPGGenerator):
    """CPG generator with full connectivity and enforced left-right symmetry.

    Like :class:`FullyConnectedCPGGenerator`, this generator creates a
    fully connected oscillator network.  The key difference is in
    ``outputs_to_actions``: the outputs of the two symmetric arm pairs are
    mirrored (with alternating sign negation) so that the robot moves
    symmetrically without requiring the genetic optimizer to discover
    symmetric gaits from scratch.
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

    def outputs_to_actions(self, _outputs: jarr, configuration: Configuration) -> jarr:
        """Map CPG outputs to joint actions while enforcing bilateral symmetry.

        The leading arm (index 0) and the two front arms (indices 1 and 4)
        use the outputs directly.  The two back arms (indices 2 and 3) mirror
        the front arms with alternating sign negation applied to even-indexed
        joints within each arm.

        Args:
            _outputs: Oscillator output array, shape ``(N,)``.
            configuration: Unused.

        Returns:
            Joint action array of the same shape as ``_outputs`` with
            symmetry enforced.
        """

        def negate_even_indices(a):
            indices = jnp.arange(len(a))
            mask = indices % 2 == 0
            return jnp.where(mask, -a, a)

        outputs = jnp.zeros_like(_outputs).astype(_outputs.dtype)

        # leading arm
        outputs = outputs.at[jnp.arange(len(_outputs) // 5 * 0, len(_outputs) // 5 * 1)].set(
            _outputs[jnp.arange(len(_outputs) // 5 * 0, len(_outputs) // 5 * 1)]
        )

        # front arms
        outputs = outputs.at[jnp.arange(len(_outputs) // 5 * 1, len(_outputs) // 5 * 2)].set(
            _outputs[jnp.arange(len(_outputs) // 5 * 1, len(_outputs) // 5 * 2)]
        )
        outputs = outputs.at[jnp.arange(len(_outputs) // 5 * 4, len(_outputs) // 5 * 5)].set(
            negate_even_indices(
                _outputs[jnp.arange(len(_outputs) // 5 * 1, len(_outputs) // 5 * 2)]
            )
        )

        # back arms
        outputs = outputs.at[jnp.arange(len(_outputs) // 5 * 2, len(_outputs) // 5 * 3)].set(
            _outputs[jnp.arange(len(_outputs) // 5 * 2, len(_outputs) // 5 * 3)]
        )
        outputs = outputs.at[jnp.arange(len(_outputs) // 5 * 3, len(_outputs) // 5 * 4)].set(
            negate_even_indices(
                _outputs[jnp.arange(len(_outputs) // 5 * 2, len(_outputs) // 5 * 3)]
            )
        )

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
