import time

import configs.subcontrollers.controller_configurations
import configs.subcontrollers.cpg_configurations
import configs.subcontrollers.genetic_configurations
import configs.subcontrollers.logger
import configs.subcontrollers.random_configurations
import configs.subcontrollers.simulation_configurations
from configs.config import Configuration
from src.controller import Input
from src.environment import Environment


def main():
    start = time.time()

    configuration = Configuration(
        configs.subcontrollers.logger.standard(),
        configs.subcontrollers.simulation_configurations.standard(),
        configs.subcontrollers.cpg_configurations.fully_connected(),
        configs.subcontrollers.random_configurations.standard(),
        configs.subcontrollers.genetic_configurations.short(),
        configs.subcontrollers.controller_configurations.body_direction()
    )
    configuration.logger.log_configuration()

    genetic_optimizer = configuration.genetic.genetic_optimizer
    starting_population = genetic_optimizer.initialize_population(
        configuration.genetic.population_size,
        configuration.controller.controller.genome_size(configuration),
        configuration.random.split()
    )
    selections, evaluations = genetic_optimizer.generation(
        configuration.controller.controller.evaluator(configuration),
        starting_population,
        configuration.genetic.number_of_generations,
        configuration.random.split(),
        configuration.logger
    )
    controller = configuration.controller.controller.train_controller(selections, evaluations, configuration)
    controller.save_controller(configuration.logger)

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)
    env_state = env.reset(configuration.random.split())
    # cpg_state = cpg_generator.modulate_cpg(cpg.reset(), 2, .3)
    cpg_state = cpg.reset()

    frames = []
    while not (env_state.terminated | env_state.truncated):
        cpg_state = controller.act(cpg_state, Input.LEFT, configuration)
        cpg_state = cpg.step(cpg_state)
        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(actions, env_state)
        frames.append(env.render(env_state))

    configuration.logger.log_video(frames, "video.mp4")

    end = time.time()
    configuration.logger.log(str(end - start))


if __name__ == '__main__':
    main()
