from abc import ABC, abstractmethod
from typing import Callable

from configs.config import Configuration
from configs.subcontrollers.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


class Controller(ABC):
    """
    The controller implements all function that determine how the brittle star behaves for user input.
    This includes the training function for the genetic algorithm (with genome size) and functions to
    train the controller from a genetic selection.
    The controller also has functions to write itself to or read itself from files.
    """

    @abstractmethod
    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
    ) -> CPGState:
        pass

    @abstractmethod
    def train_controller(
        self,
        genetic_selections: jarr,
        genetic_evaluations: jarr,
        configuration: Configuration,
    ):
        pass

    @staticmethod
    @abstractmethod
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], jarr]:
        pass

    @abstractmethod
    def genome_size(self, configuration: Configuration) -> int:
        pass

    @abstractmethod
    def save_controller(self, logger: Logger):
        pass

    @abstractmethod
    def read_controller(self, path: str):
        pass
