from abc import ABC

from configs.subcontrollers.config import Configuration
from src.cpg.cpg import CPGState
from src.jax_extra import jarr


class Controller(ABC):
    def act(self, cpg_state: CPGState, control_input: str, configuration: Configuration) -> CPGState:
        pass


class StandardController(Controller):
    def __init__(self, body_cpg: jarr):
        self.body_cpg = body_cpg

    def act(self, cpg_state: CPGState, control_input: str, configuration: Configuration):
        cpg_generator = configuration.cpg.cpg_generator
        return cpg_generator.modulate_body(cpg_state, self.body_cpg)
