from abc import abstractmethod, ABC
from typing import Callable

import jax
import jax.numpy as jnp

from configs.subconfiguration import SubConfiguration
from configs.config import Configuration
from src.controller import Controller, StandardController, Input, BodyDirectionController
from src.environment import Environment
from src.jax_extra import jarr


class ControllerConfiguration(SubConfiguration, ABC):
    """
    # todo move abstract methods to controller.py
    This class contains the configuration determining which brittle star robot controller is used.
    """

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr,
                         configuration: Configuration) -> Controller:
        pass

    @staticmethod
    @abstractmethod
    def evaluator(configuration: Configuration) -> Callable[[jarr], jarr]:
        pass

    @property
    @abstractmethod
    def genome_size(self) -> int:
        pass


def standard():
    class StandardControllerConfiguration(ControllerConfiguration):
        def __init__(self):
            super().__init__("standard")

        def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr,
                             configuration: Configuration) -> Controller:
            return StandardController(genetic_selections[0])

        @staticmethod
        def evaluator(configuration: Configuration) -> Callable[[jarr], jarr]:
            rng = configuration.random.split()

            def evaluator(arr: jarr) -> jarr:
                def _evaluator(_arr: jarr, _rng: jarr) -> jarr:
                    env = Environment(configuration)
                    env_state = env.reset(_rng)
                    cpg_generator = configuration.cpg.cpg_generator
                    cpg = cpg_generator.generate(configuration)
                    cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)
                    score = 0
                    for i in range(400):
                        cpg_state = cpg.step(cpg_state)
                        env_state = env.step(
                            cpg_generator.outputs_to_actions(cpg_state.outputs, configuration),
                            env_state
                        )
                        score += env_state.observations["disk_position"][0]
                    return score

                return jax.vmap(_evaluator)(arr, jax.random.split(rng, len(arr)))

            return evaluator

        @property
        def genome_size(self):
            cpg_generator = self._configuration.cpg.cpg_generator
            cpg = cpg_generator.generate(self._configuration)
            return cpg_generator.body_to_jarr(cpg.reset()).size

    return StandardControllerConfiguration()


def body_direction():
    class BodyDirectionControllerConfiguration(ControllerConfiguration):
        def __init__(self):
            super().__init__("body_direction")

        def train_controller(self, genetic_selections: jarr, genetic_evaluations: jarr,
                             configuration: Configuration) -> Controller:
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

            directions = [Input.RIGHT, Input.LEFT, Input.UP, Input.DOWN, Input.TURN_LEFT, Input.TURN_RIGHT]
            cpg_map = dict()
            for i, direction in enumerate(directions):
                max_index = max(range(len(all_scores)), key=lambda j: all_scores[j][i])
                cpg_map[direction] = genetic_selections[max_index]

            return BodyDirectionController(cpg_map)

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
                scores = jnp.zeros(len(all_scores))

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

        @property
        def genome_size(self):
            cpg_generator = self._configuration.cpg.cpg_generator
            cpg = cpg_generator.generate(self._configuration)
            return cpg_generator.body_to_jarr(cpg.reset()).size

    return BodyDirectionControllerConfiguration()
