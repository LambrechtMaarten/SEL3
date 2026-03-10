from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
import jax.numpy as jnp

from configs.config import Configuration
from configs.subcontrollers.logger import Logger
from src.cpg.cpg import CPGState
from src.jax_extra import jarr


class Input(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UPLEFT = "upleft"
    UPRIGHT = "upright"
    DOWNLEFT = "downleft"
    DOWNRIGHT = "downright"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class Controller(ABC):
    @abstractmethod
    def act(self, cpg_state: CPGState, control_input: Input, configuration: Configuration) -> CPGState:
        pass

    @abstractmethod
    def save_controller(self, logger: Logger):
        pass


class StandardController(Controller):
    def __init__(self, body_cpg: jarr):
        self.body_cpg = body_cpg

    def act(self, cpg_state: CPGState, control_input: Input, configuration: Configuration):
        cpg_generator = configuration.cpg.cpg_generator
        return cpg_generator.modulate_body(cpg_state, self.body_cpg)

    def save_controller(self, logger: Logger):
        pass


class BodyDirectionController(Controller):
    def __init__(self, cpg_map: Dict[Input, jarr]):
        self.cpg_map = cpg_map

    def act(self, cpg_state: CPGState, control_input: Input, configuration: Configuration):
        cpg_generator = configuration.cpg.cpg_generator
        return cpg_generator.modulate_body(cpg_state, self.cpg_map[control_input])

    def save_controller(self, logger: Logger):
        for direction in [Input.RIGHT, Input.LEFT, Input.UP, Input.DOWN, Input.TURN_LEFT, Input.TURN_RIGHT]:
            logger.log_controller(jnp.array_str(self.cpg_map[direction]))
