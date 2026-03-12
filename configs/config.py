from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.subcontrollers.logger.logger import Logger
    from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
    from configs.subcontrollers.random.random_configurations import RandomConfiguration
    from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
    from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
    from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration


@dataclass
class Configuration:
    """
    This class contains the configuration of the entire process of training a brittle star robot.
    The algorithm is meant to be unchangable, with the only way of changing training being to pass a different Configuration.
    """
    logger: "Logger"
    simulation: "SimulationConfiguration"
    cpg: "CPGConfiguration"
    random: "RandomConfiguration"
    genetic: "GeneticConfiguration"
    controller: "ControllerConfiguration"

    def __post_init__(self):
        for f in fields(self):
            getattr(self, f.name).set_configuration(self)
