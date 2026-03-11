from configs.subcontrollers.logger.logger import Logger
from src.jax_extra import jarr


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
