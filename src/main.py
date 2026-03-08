import time

import jax.numpy as jnp

import configs.subcontrollers.controller_configurations
import configs.subcontrollers.cpg_configurations
import configs.subcontrollers.genetic_configurations
import configs.subcontrollers.logger
import configs.subcontrollers.random_configurations
import configs.subcontrollers.simulation_configurations
from configs.subcontrollers.config import Configuration
from src.env import Environment


def main():
    configuration = Configuration(
        configs.subcontrollers.logger.standard(),
        configs.subcontrollers.simulation_configurations.standard(),
        configs.subcontrollers.cpg_configurations.standard(),
        configs.subcontrollers.random_configurations.standard(),
        configs.subcontrollers.genetic_configurations.standard(),
        configs.subcontrollers.controller_configurations.standard()
    )

    configuration.logger.log_configuration()

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)

    configuration.logger.log_genetic_generation(jnp.zeros(5))
    configuration.logger.log_genetic_generation(jnp.zeros(5))

    genetic_optimizer = configuration.genetic.genetic_optimizer

    starting_population = genetic_optimizer.initialize_population(
        configuration.genetic.population_size,
        cpg_generator.body_to_jarr(cpg.reset()).size,
        configuration.random.split()
    )
    selections, evaluations = genetic_optimizer.generation(
        configuration.controller.evaluator,
        starting_population,
        configuration.genetic.generations,
        configuration.random.split(),
        configuration,
    )
    controller = configuration.controller.train_controller(selections, evaluations)

    frames = []
    env_state = env.reset()
    cpg_state = cpg.reset()
    while not (env_state.terminated | env_state.truncated):
        cpg_state = controller.act(cpg_state, "todo, input", configuration)
        cpg_state = cpg.step(cpg_state)
        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(actions, env_state)
        frames.append(env.render(env_state))

    configuration.logger.log_video(frames, "video.mp4")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
