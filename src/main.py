import time

from configs.config import Configuration
from configs.subconfiguration import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.register import register
from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration
from src.controller import Input
from src.environment import Environment


def main():
    register()
    start = time.time()

    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "standard"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "fully_connected"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "short"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "body_direction")
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
    configuration.controller.controller.train_controller(selections, evaluations, configuration)
    configuration.controller.controller.save_controller(configuration.logger)

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)
    env_state = env.reset(configuration.random.split())
    # cpg_state = cpg_generator.modulate_cpg(cpg.reset(), 2, .3)
    cpg_state = cpg.reset()

    frames = []
    while not (env_state.terminated | env_state.truncated):
        cpg_state = configuration.controller.controller.act(cpg_state, Input.LEFT, configuration)
        cpg_state = cpg.step(cpg_state)
        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(actions, env_state)
        frames.append(env.render(env_state))

    configuration.logger.log_video(frames, "video.mp4")

    end = time.time()
    configuration.logger.log(str(end - start))


if __name__ == '__main__':
    main()
