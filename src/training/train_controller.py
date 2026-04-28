import os
import time

import jax

from configs.config import Configuration
from src.controller.control_input import ControlInput
from src.environment.environment import Environment


def train_controller(configuration: Configuration):
    """Trains a brittle star controller using the configured genetic algorithm.

    Runs the full training pipeline:
        1. Initializes the logger and saves the configuration.
        2. Creates an initial population, either random or warm-started
           from pre-trained weights if available.
        3. Runs the genetic optimization loop for the configured number
           of generations.
        4. Converts the best selections into a deployable controller.
        5. Saves the trained controller via the logger.
        6. Renders a video of the trained controller unless running
           in headless mode.
        7. Logs the total training time.

    Args:
        configuration: The full training configuration, including logger,
            simulation, CPG, random, genetic, and controller settings.
    """
    start = time.time()

    configuration.logger.init_logger()
    configuration.logger.log_configuration()

    genetic_optimizer = configuration.genetic.genetic_optimizer
    controller = configuration.controller.controller

    # Gebruik pre-getrainde gewichten als startpunt als die beschikbaar zijn
    if hasattr(controller, 'weights') and controller.weights is not None:
        print("Warm start: populatie geïnitialiseerd rond pre-getrainde gewichten")
        rng = configuration.random.split()
        noise = jax.random.normal(rng, (
            configuration.genetic.population_size,
            controller.weights.size
        )) * 0.01
        starting_population = controller.weights[None, :] + noise
    else:
        starting_population = genetic_optimizer.initialize_population(
            configuration.genetic.population_size,
            controller.genome_size(configuration),
            configuration.random.split(),
        )

    evaluator_fn = controller.evaluator(
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
    controller.train_controller(
        selections, evaluations, configuration
    )
    controller.save_controller(configuration.logger)

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)
    env_state = env.reset(configuration.random.split())
    cpg_state = cpg.reset()

    headless = os.getenv("HEADLESS")

    if not headless:
        frames = []
        while not (env_state.terminated | env_state.truncated):
            cpg_state = controller.act(
                cpg_state, ControlInput.LEFT, configuration, env_state
            )
            cpg_state = cpg.step(cpg_state)
            actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions, env_state)
            frames.append(env.render(env_state))

        configuration.logger.log_video(frames, "video.mp4")

    end = time.time()
    configuration.logger.log("time", str(end - start))
