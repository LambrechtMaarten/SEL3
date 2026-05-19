from typing import Any, Callable, Tuple

import jax
from jax import numpy as jnp
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from configs.subconfigurations.logger.logger import Logger
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

    @staticmethod
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], Tuple[jarr, jarr]]:
        def evaluator(arr: jarr) -> Tuple[jarr, jarr]:
            env = Environment(configuration)

            def _evaluator(_arr: jarr, _rng: jarr) -> tuple[Any, Any, Any]:
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)

                max_steps = 800

                def step_fn(_, val):
                    cpg_state, env_state, energy = val
                    cpg_state = cpg.step(cpg_state)
                    actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
                    max_force = env.action_space.high[0]
                    energy += jnp.sum(
                        jnp.pow(max_force - jnp.clip(jnp.abs(actions), a_max=max_force), 2)
                    )
                    env_state = env.step(
                        actions,
                        env_state,
                    )
                    return cpg_state, env_state, energy

                cpg_state, env_state, energy = jax.lax.fori_loop(
                    0, max_steps, step_fn, (cpg_state, env_state, 0)
                )
                return (
                    energy,
                    env_state.observations["disk_position"][0],
                    env_state.observations["disk_position"][1],
                )

            new_rngs = jax.random.split(rng, len(arr))
            energies, x_poss, y_poss = jax.vmap(_evaluator)(arr, new_rngs)
            scores = energies

            groups = jnp.floor(x_poss * 5).astype(int)
            group_sizes = jnp.count_nonzero(
                jnp.equal(
                    groups,
                    jnp.repeat(groups, groups.shape[0]).reshape((groups.shape[0], groups.shape[0])),
                ),
                axis=0,
            )
            max_scores_in_group = jnp.max(
                jnp.where(groups[:, None] == groups[None, :], scores, -jnp.inf), axis=1
            )
            rescaled_scores = scores / max_scores_in_group
            rescaled_scores = rescaled_scores * (0.99**group_sizes)
            # jax.debug.log("{y}", y=groups)
            # jax.debug.log("{y}", y=scores)

            return rescaled_scores, scores

        return evaluator

    @staticmethod
    def get_edges(configuration: Configuration, rng) -> Callable[[jarr], Tuple[jarr, jarr]]:
        def evaluator(arr: jarr) -> Tuple[jarr, jarr]:
            env = Environment(configuration)

            def _evaluator(_arr: jarr, _rng: jarr) -> tuple[Any, Any, Any, Any]:
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)

                max_steps = 800

                def step_fn(_, val):
                    i, cpg_state, env_state, energy, edges = val
                    cpg_state = cpg.step(cpg_state)
                    actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
                    energy += jnp.sum(
                        (
                            env.action_space.high[0]
                            - jnp.clip(jnp.abs(actions), a_max=env.action_space.high[0])
                        )
                        ** 2
                    )
                    env_state = env.step(
                        actions,
                        env_state,
                    )

                    return (
                        i + 1,
                        cpg_state,
                        env_state,
                        energy,
                        edges.at[i].set(env_state.observations["joint_position"]),
                    )

                i, cpg_state, env_state, energy, edges = jax.lax.fori_loop(
                    0,
                    max_steps,
                    step_fn,
                    (
                        0,
                        cpg_state,
                        env_state,
                        0,
                        jnp.broadcast_to(
                            env_state.observations["joint_position"],
                            (800,) + env_state.observations["joint_position"].shape,
                        ),
                    ),
                )
                return (
                    energy,
                    env_state.observations["disk_position"][0],
                    env_state.observations["disk_position"][1],
                    edges,
                )

            new_rngs = jax.random.split(rng, len(arr))
            energies, x_poss, y_poss, edges = jax.vmap(_evaluator)(arr, new_rngs)
            groups = jnp.floor(x_poss * 5).astype(int)

            return groups, edges

        return evaluator

    def genome_size(self, configuration: Configuration) -> int:
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        return cpg_generator.body_to_jarr(cpg.reset()).size

    def save_controller(self, logger: Logger, name: str = "controller"):
        # Standard logger needs string, but wandb does not :(
        # noinspection PyTypeChecker
        logger.log_controller(jnp.array_str(self.body_cpg))

    def read_controller(self, path: str):
        with open(path, "r") as f:
            arrays = f.read()
            self.body_cpg = jnp.array(
                [float(x) for x in arrays.replace("[", " ").replace("]", " ").split()]
            )
