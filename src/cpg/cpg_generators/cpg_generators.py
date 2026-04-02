from abc import ABC, abstractmethod

import jax.numpy as jnp

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class CPGGenerator(ABC):
    """
    This class represents a specific CPG configuration, specifying how oscilators map to actuarors
    """

    @abstractmethod
    def generate(self, configuration: Configuration) -> CPG:
        pass

    @abstractmethod
    def outputs_to_actions(self, outputs: jarr, configuration: Configuration) -> jarr:
        pass

    @abstractmethod
    def modulate_symmetric_rotation(self, cpg_state: CPGState, clockwise_rotations: int) -> CPGState:
        pass

    @staticmethod
    def modulate_body(cpg_state: CPGState, body: jarr) -> CPGState:
        i = 0
        frequency = body[i:i + 1][0]
        i += 1
        amplitude_goals = body[i:i + cpg_state.amplitude_goals.size]
        i += cpg_state.amplitude_goals.size
        offset_goals = body[i:i + cpg_state.offset_goals.size]
        i += cpg_state.offset_goals.size
        coupled_phase_biases = body[i:].reshape(cpg_state.coupled_phase_biases.shape)

        # noinspection PyUnresolvedReferences
        return cpg_state.replace(
            frequency=frequency,
            amplitude_goals=amplitude_goals,
            offset_goals=offset_goals,
            coupled_phase_biases=coupled_phase_biases
        )

    @staticmethod
    def body_to_jarr(cpg_state: CPGState) -> jarr:
        return jnp.concatenate([
            jnp.atleast_1d(cpg_state.frequency),
            cpg_state.amplitude_goals,
            cpg_state.offset_goals,
            cpg_state.coupled_phase_biases.ravel()
        ]).flatten()
