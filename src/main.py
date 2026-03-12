import sys

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.register import register
from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration
from src.simulation.simulate_controller import simulate_controller
from src.training.train_controller import train_controller


def create_configuration():
    return Configuration(
        SubConfigurationMap.get_configuration(Logger, "standard"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "short"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "standard")
    )


if __name__ == '__main__':
    register()
    if sys.argv[1] == "train":
        train_controller(create_configuration())
    elif sys.argv[1] == "simulate":
        simulate_controller(sys.argv[2])
