"""Abstract base class for CPG generator implementations."""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class CPGGenerator(ABC):
    """Abstract factory and utilities for a specific CPG oscillator topology.

    A CPGGenerator defines how oscillators are wired together (via the
    adjacency matrix), how their outputs are mapped to joint actions, and how
    the CPG state is permuted when the robot rotates symmetrically.
    """

    @abstractmethod
    def generate(self, configuration: Configuration) -> CPG:
        """Construct and return a CPG instance for the given configuration.

        Args:
            configuration: Global simulation and training configuration.

        Returns:
            A :class:`CPG` instance ready to be reset and stepped.
        """

    @abstractmethod
    def outputs_to_actions(self, outputs: jarr, configuration: Configuration) -> jarr:
        """Map CPG oscillator outputs to joint torque actions.

        Args:
            outputs: Oscillator output array, shape ``(N,)``.
            configuration: Global simulation and training configuration.

        Returns:
            Joint action array suitable for stepping the MuJoCo environment.
        """

    @abstractmethod
    def modulate_symmetric_rotation(
        self, cpg_state: CPGState, clockwise_rotations: int
    ) -> CPGState:
        """Permute the CPG state to account for a clockwise rotation of the leading arm.

        Args:
            cpg_state: CPG state to be rotated.
            clockwise_rotations: Number of arm positions to rotate clockwise.

        Returns:
            Permuted :class:`CPGState` corresponding to the rotated arm
            assignment.
        """

    @staticmethod
    def modulate_body(cpg_state: CPGState, body: jarr) -> CPGState:
        """Apply a flat genome array to a CPG state, updating its parameters.

        The genome is expected to be laid out as:
        ``[frequency (1), amplitude_goals (N), offset_goals (N),
        coupled_phase_biases (N*N)]``.

        Args:
            cpg_state: CPG state whose shape metadata is used to unpack the
                genome.
            body: Flat genome array produced by :meth:`body_to_jarr`.

        Returns:
            A new :class:`CPGState` with frequency, amplitude goals, offset
            goals, and phase biases set from the genome.
        """
        i = 0
        frequency = body[i : i + 1][0]
        i += 1
        amplitude_goals = body[i : i + cpg_state.amplitude_goals.size]
        i += cpg_state.amplitude_goals.size
        offset_goals = body[i : i + cpg_state.offset_goals.size]
        i += cpg_state.offset_goals.size
        coupled_phase_biases = body[i:].reshape(cpg_state.coupled_phase_biases.shape)

        # noinspection PyUnresolvedReferences
        return cpg_state.replace(
            frequency=frequency,
            amplitude_goals=amplitude_goals,
            offset_goals=offset_goals,
            coupled_phase_biases=coupled_phase_biases,
        )

    @staticmethod
    def body_to_jarr(cpg_state: CPGState) -> jarr:
        """Serialise the CPG body parameters from a state into a flat array.

        The array layout mirrors what :meth:`modulate_body` expects:
        ``[frequency, amplitude_goals, offset_goals,
        coupled_phase_biases.ravel()]``.

        Args:
            cpg_state: CPG state containing the parameters to serialise.

        Returns:
            Flat JAX array representing the full CPG body genome.
        """
        return jnp.concatenate(
            [
                jnp.atleast_1d(cpg_state.frequency),
                cpg_state.amplitude_goals,
                cpg_state.offset_goals,
                cpg_state.coupled_phase_biases.ravel(),
            ]
        ).flatten()
