import os
import time

import jax

from configs.config import Configuration
from src.controller.control_input import ControlInput
from src.environment.environment import Environment


def train_controller(configuration: Configuration):
    start = time.time()

    configuration.logger.init_logger()
    configuration.logger.log_configuration()

    genetic_optimizer = configuration.genetic.genetic_optimizer
    starting_population = genetic_optimizer.initialize_population(
        configuration.genetic.population_size,
        configuration.controller.controller.genome_size(configuration),
        configuration.random.split(),
    )
    evaluator_fn = configuration.controller.controller.evaluator(
        configuration, configuration.random.split()
    )
    evaluator_fn = jax.jit(evaluator_fn)
    selections, evaluations = genetic_optimizer.generation(
        evaluator_fn,
        starting_population,
        configuration.genetic.number_of_generations,
        configuration.random.split(),
        configuration.logger,
    )
    configuration.controller.controller.train_controller(
        selections, evaluations, configuration
    )
    configuration.controller.controller.save_controller(configuration.logger)

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)
    env_state = env.reset(configuration.random.split())
    cpg_state = cpg.reset()

    headless = os.getenv("HEADLESS")

    if not headless:
        frames = []
        while not (env_state.terminated | env_state.truncated):
            cpg_state = configuration.controller.controller.act(
                cpg_state, ControlInput.LEFT, configuration
            )
            cpg_state = cpg.step(cpg_state)
            actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions, env_state)
            # Skip rendering on HPC
            frames.append(env.render(env_state))

        configuration.logger.log_video(frames, "video.mp4")

    end = time.time()
    configuration.logger.log(str(end - start))
