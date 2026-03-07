from abc import abstractmethod, ABC

from configs.subconfiguration import SubConfiguration
from src.jax_extra import jarr


class ControllerConfiguration(SubConfiguration, ABC):
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def train_controller(self, genetic_selections: jarr):
        pass


def standard():
    class StandardController(ControllerConfiguration):
        def __init__(self):
            super().__init__("standard")

        def train_controller(self, genetic_selections: jarr):
            pass

    return StandardController()
