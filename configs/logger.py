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
    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def log_configuration(self):
        pass

    @abstractmethod
    def log_genetic_generation(self, generation: jarr):
        pass

    @abstractmethod
    def log_video(self, frames, name):
        pass


def silent():
    class SilentLogger(Logger):
        def __init__(self):
            super().__init__("silent_logger")

        def log_configuration(self):
            pass

        def log_genetic_generation(self, generation: jarr):
            pass

        def log_video(self, frames, name):
            pass

    return SilentLogger()


def standard():
    class StandardLogger(Logger):
        def __init__(self):
            super().__init__("standard")
            self.base_folder = os.path.join("..", "output", datetime.datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))
            if not os.path.exists(self.base_folder):
                os.makedirs(self.base_folder)

        def log_configuration(self):
            data = {}
            for f in fields(self.configuration):
                data[f.name] = getattr(self.configuration, f.name).name

            json_str = json.dumps(data, indent=4)
            with open(os.path.join(self.base_folder, "configuration.json"), "w") as f:
                f.write(json_str)

        def log_genetic_generation(self, generation: jarr):
            path = Path(os.path.join(self.base_folder, "genetic"))
            if not os.path.exists(path):
                path.touch()
            with open(path, "a") as f:
                f.write(str(generation))

        def log_video(self, frames, name):
            save_video(frames, os.path.join(self.base_folder, name))

    return StandardLogger()
