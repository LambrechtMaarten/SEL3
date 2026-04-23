from configs.subconfigurations.logger.logger import Logger
from src.jax_extra.jax_extra import jarr


class SilentLogger(Logger):
    """
    This Logger implementation does not log anything.
    Usefull for the simulation or for debugging.
    """

    def __init__(self):
        super().__init__("silent_logger")

    def init_logger(self):
        pass

    def log_configuration(self):
        pass

    def log_genetic_generation(self, population: jarr, selections: jarr, evaluations: jarr):
        pass

    def log_video(self, frames, name):
        pass

    def log(self, logging: str):
        pass
