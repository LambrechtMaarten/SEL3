import os
import time

import jax
from fontTools.ufoLib import convertFontInfoValueForAttributeFromVersion2ToVersion1

from configs.config import Configuration
from configs.subconfigurations.controller.controller_configurations import ControllerConfiguration
from src.controller.control_input import ControlInput
from src.controller.one_direction_map_elites_controller import OneDirectionMapElitesController
from src.environment.environment import Environment


def train_archive(configuration: Configuration):
    configuration.controller = ControllerConfiguration("map elites", OneDirectionMapElitesController())
    configuration.controller.set_configuration(configuration)

    configuration.logger.init_logger()
    configuration.logger.log_configuration()

    genetic_optimizer = configuration.genetic.genetic_optimizer
    controller = configuration.controller.controller

    starting_population = genetic_optimizer.initialize_population(
        configuration.genetic.population_size,
        controller.genome_size(configuration),
        configuration.random.split(),
    )

    evaluator_fn = jax.jit(controller.evaluator(configuration, configuration.random.split()))
    selections, evaluations = genetic_optimizer.generation(
        evaluator_fn,
        starting_population,
        configuration.genetic.number_of_generations,
        configuration.random.split(),
        configuration.logger,
    )

    controller: OneDirectionMapElitesController = configuration.controller.controller
    get_edges = jax.jit(controller.get_edges(configuration, configuration.random.split()))
    x_positions, groups, edges = get_edges(selections)

    for i in range(len(x_positions)):
        env = Environment(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        env_state = env.reset(configuration.random.split())
        cpg_state = cpg.reset()
        frames = []
        while not (env_state.terminated | env_state.truncated):
            cpg_state = configuration.cpg.cpg_generator.modulate_body(cpg_state, selections[i])
            cpg_state = cpg.step(cpg_state)
            actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions, env_state)
            frames.append(env.render(env_state))
        configuration.logger.log_video(frames, f'video_{groups[i]}.mp4')


