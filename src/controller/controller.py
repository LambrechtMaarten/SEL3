"""Abstract base class defining the interface for all brittle star robot controllers."""

from abc import ABC, abstractmethod
from typing import Callable

from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from configs.subconfigurations.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class Controller(ABC):
    """Abstract base class for all brittle star robot controllers.

    A controller determines how the brittle star behaves in response to user
    input. Concrete subclasses must implement the locomotion logic, a fitness
    evaluator for the genetic algorithm, serialization, and deserialization.
    """

    @abstractmethod
    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
        env_state: BaseEnvState,
    ) -> CPGState:
        """Compute joint actions for the current simulation step.

        Args:
            cpg_state: Current state of the Central Pattern Generator.
            control_input: High-level locomotion command (angle and speed).
            configuration: Global simulation and training configuration.
            env_state: Current MuJoCo environment state including observations.

        Returns:
            Updated CPG state or a JAX array of joint torques, depending on
            the concrete controller implementation.
        """

    @staticmethod
    @abstractmethod
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], jarr]:
        """Return a fitness evaluator function for use with genetic optimization.

        The returned callable maps a batch of genomes (JAX arrays) to scalar
        fitness scores.

        Args:
            configuration: Global simulation and training configuration.
            rng: JAX random key used to seed environment resets.

        Returns:
            A callable that accepts a genome array and returns fitness scores.
        """

    @abstractmethod
    def genome_size(self, configuration: Configuration) -> int:
        """Return the number of parameters (genes) in a single genome.

        Args:
            configuration: Global simulation and training configuration.

        Returns:
            Integer genome length expected by the genetic optimizer.
        """

    @abstractmethod
    def save_controller(self, logger: Logger, name: str = "controller"):
        """Persist the controller's learned parameters via the logger.

        Args:
            logger: Logger instance used to write controller data to disk or
                a remote tracking service.
            name: Filename stem used when writing to disk.
        """

    @abstractmethod
    def read_controller(self, path: str):
        """Load controller parameters from a previously saved file.

        Args:
            path: Filesystem path to the saved controller file.
        """
