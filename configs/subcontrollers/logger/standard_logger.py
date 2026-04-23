import datetime
import json
import os
from dataclasses import fields
from pathlib import Path

import jax.numpy as jnp

from configs.subcontrollers.logger.logger import Logger
from src.jax_extra.jax_extra import jarr
from src.render.render import save_video


class StandardLogger(Logger):
    """
    The standard Logger implementation.
    """

    def __init__(self):
        super().__init__("standard")

    def init_logger(self):
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

    def log_configuration(self):
        data = {}
        for f in fields(self._configuration):
            data[f.name] = getattr(self._configuration, f.name).name

        json_str = json.dumps(data, indent=4)
        with open(os.path.join(self.base_folder, "configuration.json"), "w") as f:
            f.write(json_str)

    def log_genetic_generation(
        self, population: jarr, selections: jarr, evaluations: jarr
    ):
        path = Path(os.path.join(self.base_folder, "genetic"))
        if not os.path.exists(path):
            path.touch()
        with open(path, "a") as f:
            jnp.set_printoptions(threshold=(jnp.inf))
            # noinspection PyTypeChecker
            f.write(jnp.array_str(population))
            f.write("\n")
            # noinspection PyTypeChecker
            f.write(jnp.array_str(selections))
            f.write("\n")
            # noinspection PyTypeChecker
            f.write(jnp.array_str(evaluations))
            f.write("\n")

    def log_video(self, frames, name):
        save_video(frames, str(os.path.join(self.base_folder, name)))

    def log(self, logging: str):
        path = Path(os.path.join(self.base_folder, "log"))
        if not os.path.exists(path):
            path.touch()
        with open(path, "a") as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}]")
            f.write(logging)
            f.write("\n")
