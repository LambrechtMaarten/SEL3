from typing import Callable

import jax
from jax import numpy as jnp

from configs.config import Configuration
from configs.subcontrollers.logger.logger import Logger
from src.controller.controller import Controller
from src.controller.control_input import ControlInput
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr


class BodyDirectionController(Controller):
    def __init__(self):
        self.cpg_map = dict()

    def act(self, cpg_state: CPGState, control_input: ControlInput, configuration: Configuration):
        cpg_generator = configuration.cpg.cpg_generator
        if control_input == ControlInput.ZZZ:
            return cpg_generator.modulate_body(
                cpg_state,
                cpg_generator.body_to_jarr(cpg_generator.generate(configuration).reset())
            )
        return cpg_generator.modulate_body(cpg_state, self.cpg_map[control_input])

    def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr,
                         configuration: Configuration):
        all_scores = []
        rng = configuration.random.split()
        for i in range(len(genetic_selections)):
            rng, _rng = jax.random.split(rng)
            env = Environment(configuration)
            env_state = env.reset(_rng)
            cpg_generator = configuration.cpg.cpg_generator
            cpg = cpg_generator.generate(configuration)
            cpg_state = cpg_generator.modulate_body(cpg.reset(), genetic_selections[i])
            for _ in range(200):
                cpg_state = cpg.step(cpg_state)
                env_state = env.step(
                    cpg_generator.outputs_to_actions(cpg_state.outputs, configuration),
                    env_state
                )
            all_scores.append([
                env_state.observations["disk_position"][0],
                -env_state.observations["disk_position"][0],
                env_state.observations["disk_position"][1],
                -env_state.observations["disk_position"][1],
                -env_state.observations["disk_rotation"][1],
                env_state.observations["disk_rotation"][1],
            ])

        directions = [ControlInput.RIGHT, ControlInput.LEFT, ControlInput.UP, ControlInput.DOWN, ControlInput.TURN_LEFT, ControlInput.TURN_RIGHT]
        for i, direction in enumerate(directions):
            max_index = max(range(len(all_scores)), key=lambda j: all_scores[j][i])
            self.cpg_map[direction] = genetic_selections[max_index]

    @staticmethod
    def evaluator(configuration: Configuration) -> Callable[[jarr], jarr]:
        def evaluator(arr: jarr) -> jarr:
            all_scores = []

            def _evaluator(_arr: jarr, _rng: jarr, ):
                env = Environment(configuration)
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)
                for _ in range(200):
                    cpg_state = cpg.step(cpg_state)
                    env_state = env.step(
                        cpg_generator.outputs_to_actions(cpg_state.outputs, configuration),
                        env_state
                    )
                return [
                    env_state.observations["disk_position"][0],
                    -env_state.observations["disk_position"][0],
                    env_state.observations["disk_position"][1],
                    -env_state.observations["disk_position"][1],
                    -env_state.observations["disk_rotation"][1],
                    env_state.observations["disk_rotation"][1],
                ]

            rng = configuration.random.split()
            for i in range(len(arr)):
                rng, _rng = jax.random.split(rng)
                all_scores.append(_evaluator(arr[i], _rng))
            scores: jarr = jnp.zeros(len(all_scores))

            counts = [0] * 6
            for t in all_scores:
                order = sorted(range(6), key=lambda j: t[j], reverse=True)
                for i in order:
                    if counts[i] < (len(all_scores) + 5) // 6:
                        scores = scores.at[i].set(t[i])
                        counts[i] += 1
                        break
            return scores

        return evaluator

    def genome_size(self, configuration: Configuration):
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        return cpg_generator.body_to_jarr(cpg.reset()).size

    def save_controller(self, logger: Logger):
        for direction in [ControlInput.RIGHT, ControlInput.LEFT, ControlInput.UP, ControlInput.DOWN, ControlInput.TURN_LEFT, ControlInput.TURN_RIGHT]:
            # noinspection PyTypeChecker
            logger.log_controller(jnp.array_str(self.cpg_map[direction]))

    def read_controller(self, path: str):
        with open(path, "r") as f:
            arrays = f.read()
            jnp_array = jnp.array([float(x) for x in arrays.replace("[", " ").replace("]", " ").split()])
            jnp_array = jnp.reshape(jnp_array, (6, -1))
            for i, direction in enumerate(
                    [ControlInput.RIGHT, ControlInput.LEFT, ControlInput.UP, ControlInput.DOWN, ControlInput.TURN_LEFT, ControlInput.TURN_RIGHT]):
                self.cpg_map[direction] = jnp_array[i]
