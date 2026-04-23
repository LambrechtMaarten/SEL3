from typing import Callable, Any

import jax
from jax import numpy as jnp
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from configs.subcontrollers.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr


class OneDirectionMapElitesController(Controller):
    """
    This controller sets a cpg for the entire body when it gets input, trying to get as far away from (0,0) as possible.
    """

    def __init__(self):
        self.body_cpg: jarr | None = None

    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
        env_state: BaseEnvState,
    ):
        cpg_generator = configuration.cpg.cpg_generator
        return cpg_generator.modulate_body(cpg_state, self.body_cpg)

    def train_controller(
        self,
        genetic_selections: jarr,
        genetic_evaluations: jarr,
        configuration: Configuration,
    ):
        self.body_cpg = genetic_selections[jnp.argmax(genetic_evaluations)]

    @staticmethod
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], jarr]:
        def evaluator(arr: jarr) -> jarr:
            env = Environment(configuration)

            def _evaluator(_arr: jarr, _rng: jarr) -> tuple[Any, Any]:
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)

                max_steps = 800

                def step_fn(_, val):
                    cpg_state, env_state = val
                    cpg_state = cpg.step(cpg_state)
                    env_state = env.step(
                        cpg_generator.outputs_to_actions(
                            cpg_state.outputs, configuration
                        ),
                        env_state,
                    )
                    return cpg_state, env_state

                cpg_state, env_state = jax.lax.fori_loop(
                    0, max_steps, step_fn, (cpg_state, env_state)
                )
                return env_state.observations["disk_position"][0], env_state.observations["disk_position"][1]

            new_rngs = jax.random.split(rng, len(arr))
            x_poss, y_poss = jax.vmap(_evaluator)(arr, new_rngs)
            scores = x_poss - y_poss / 2

            groups = jnp.floor(x_poss * 10).astype(int)
            max_scores = jnp.max(jnp.where(groups[:, None] == groups[None, :], scores, -jnp.inf), axis=1)
            rescaled_scores = scores / max_scores

            jax.debug.log("{x}", x=scores)
            jax.debug.log("{x}", x=rescaled_scores)
            return rescaled_scores

        return evaluator

    def genome_size(self, configuration: Configuration) -> int:
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        return cpg_generator.body_to_jarr(cpg.reset()).size

    def save_controller(self, logger: Logger):
        # Standard logger needs string, but wandb does not :(
        # noinspection PyTypeChecker
        logger.log_controller(jnp.array_str(self.body_cpg))

    def read_controller(self, path: str):
        with open(path, "r") as f:
            arrays = f.read()
            self.body_cpg = jnp.array(
                [float(x) for x in arrays.replace("[", " ").replace("]", " ").split()]
            )
