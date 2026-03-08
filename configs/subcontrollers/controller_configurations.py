from abc import abstractmethod, ABC

import jax

from configs.subconfiguration import SubConfiguration
from configs.subcontrollers.config import Configuration
from src.controller import Controller, StandardController
from src.env import Environment
from src.jax_extra import jarr


class ControllerConfiguration(SubConfiguration, ABC):
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr) -> Controller:
        pass

    @staticmethod
    @abstractmethod
    def evaluator(arr: jarr, _configuration: Configuration):
        pass


def standard():
    class StandardControllerConfiguration(ControllerConfiguration):
        def __init__(self):
            super().__init__("standard")

        def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr) -> Controller:
            return StandardController(genetic_selections[0])

        @staticmethod
        def evaluator(arr: jarr, _configuration: Configuration) -> jarr:
            def _evaluator(_arr: jarr) -> jarr:
                _env = Environment(_configuration)
                _env_state = _env.reset()
                _cpg_generator = _configuration.cpg.cpg_generator
                _cpg = _cpg_generator.generate(_configuration)
                _cpg_state = _cpg_generator.modulate_body(_cpg.reset(), _arr)
                score = 0
                for i in range(200):
                    _cpg_state = _cpg.step(_cpg_state)
                    _env_state = _env.step(_cpg_generator.outputs_to_actions(_cpg_state.outputs, _configuration),
                                           _env_state)
                    score += _env_state.info["xy_distance_from_origin"]
                return score

            return jax.vmap(_evaluator)(arr)

    return StandardControllerConfiguration()
