import datetime
import os.path
from abc import abstractmethod
from pathlib import Path

from configs.subconfiguration import SubConfiguration
from src.jax_extra.jax_extra import jarr


class Logger(SubConfiguration):
    """
    This configuration class implements all functions used to log information.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.base_folder = os.path.join(
            "output", datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S")
        )

    @abstractmethod
    def init_logger(self):
        pass

    def log_controller(self, string):
        path = Path(os.path.join(self.base_folder, "controller"))
        if not os.path.exists(path):
            path.touch()
        with open(path, "a") as f:
            f.write(string)

    @abstractmethod
    def log_configuration(self):
        pass

    @abstractmethod
    def log_genetic_generation(
        self, population: jarr, selections: jarr, evaluations: jarr, generation: int
    ):
        pass

    @abstractmethod
    def log_video(self, frames, name):
        pass

    @abstractmethod
    def log(self, name: str, logging: str):
        pass
