from typing import Callable

import jax
import jax.numpy as jnp

from configs.config import Configuration
from configs.subcontrollers.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr


class BasicSymmetryController(Controller):
    """
    This controller implements a brittle star going in one direction.
    Depending on the control input, the cpg is rotated to make use of the brittle star's symmetry.
    """

    def __init__(self):
        self.body_cpg = None

    def act(self, cpg_state: CPGState, control_input: ControlInput, configuration: Configuration) -> CPGState:
        cpg_generator = configuration.cpg.cpg_generator
        if control_input == ControlInput.ZZZ:
            return cpg_generator.modulate_body(
                cpg_state, cpg_generator.body_to_jarr(cpg_generator.generate(configuration).reset())
            )
        return cpg_generator.modulate_body(cpg_state, self.body_cpg)

    def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr, configuration: Configuration):
        self.body_cpg = genetic_selections[0]

    @staticmethod
    def evaluator(configuration: Configuration) -> Callable[[jarr], jarr]:
        rng = configuration.random.split()

        def evaluator(arr: jarr) -> jarr:
            def _evaluator(_arr: jarr, _rng: jarr) -> jarr | float:
                env = Environment(configuration)
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)
                for i in range(1000):
                    cpg_state = cpg.step(cpg_state)
                    env_state = env.step(
                        cpg_generator.outputs_to_actions(cpg_state.outputs, configuration),
                        env_state
                    )
                return env_state.info["xy_distance_from_origin"]

            return jax.vmap(_evaluator)(arr, jax.random.split(rng, len(arr)))

        return evaluator

    def genome_size(self, configuration: Configuration) -> int:
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        return cpg_generator.body_to_jarr(cpg.reset()).size

    def save_controller(self, logger: Logger):
        logger.log_controller(jnp.array_str(self.body_cpg))

    def read_controller(self, path: str):
        with open(path, "r") as f:
            arrays = f.read()
            self.body_cpg = jnp.array([float(x) for x in arrays.replace("[", " ").replace("]", " ").split()])
