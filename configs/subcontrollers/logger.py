import os.path
from abc import abstractmethod
from dataclasses import fields
import datetime
import json
from pathlib import Path

from dm_control.suite.dog import fetch

from configs.subconfiguration import SubConfiguration
from src.jax_extra import jarr
from src.render import save_video


class Logger(SubConfiguration):
    """
    This configuration class implements all functions used to log information.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.base_folder = os.path.join("..", "output", datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

    def log_controller(self, str):
        path = Path(os.path.join(self.base_folder, "controller"))
        if not os.path.exists(path):
            path.touch()
        with open(path, "a") as f:
            f.write(str)

    @abstractmethod
    def log_configuration(self):
        pass

    @abstractmethod
    def log_genetic_generation(self, population: jarr, selections: jarr, evaluations: jarr):
        pass

    @abstractmethod
    def log_video(self, frames, name):
        pass

    @abstractmethod
    def log(self, logging: str):
        pass


def silent():
    class SilentLogger(Logger):
        def __init__(self):
            super().__init__("silent_logger")

        def log_configuration(self):
            pass

        def log_genetic_generation(self, population: jarr, selections: jarr, evaluations: jarr):
            pass

        def log_video(self, frames, name):
            pass

        def log(self, logging: str):
            pass

    return SilentLogger()


def standard():
    class StandardLogger(Logger):
        def __init__(self):
            super().__init__("standard")
            if not os.path.exists(self.base_folder):
                os.makedirs(self.base_folder)

        def log_configuration(self):
            data = {}
            for f in fields(self._configuration):
                data[f.name] = getattr(self._configuration, f.name).name

            json_str = json.dumps(data, indent=4)
            with open(os.path.join(self.base_folder, "configuration.json"), "w") as f:
                f.write(json_str)

        def log_genetic_generation(self, population: jarr, selections: jarr, evaluations: jarr):
            path = Path(os.path.join(self.base_folder, "genetic"))
            if not os.path.exists(path):
                path.touch()
            with open(path, "a") as f:
                import jax.numpy as jnp
                f.write(jnp.array_str(population))
                f.write("\n")
                f.write(jnp.array_str(selections))
                f.write("\n")
                f.write(jnp.array_str(evaluations))
                f.write("\n")

        def log_video(self, frames, name):
            save_video(frames, os.path.join(self.base_folder, name))

        def log(self, logging: str):
            path = Path(os.path.join(self.base_folder, "log"))
            if not os.path.exists(path):
                path.touch()
            with open(path, "a") as f:
                f.write(f'[{datetime.datetime.now().strftime("%H:%M:%S")}]')
                f.write(logging)
                f.write("\n")

    return StandardLogger()
