from typing import TYPE_CHECKING
from dataclasses import dataclass, fields

if TYPE_CHECKING:
    from configs.subcontrollers.logger import Logger
    from configs.subcontrollers.cpg_configurations import CPGConfiguration
    from configs.subcontrollers.random_configurations import RandomConfiguration
    from configs.subcontrollers.genetic_configurations import GeneticConfiguration
    from configs.subcontrollers.controller_configurations import ControllerConfiguration
    from configs.subcontrollers.simulation_configurations import SimulationConfiguration


@dataclass
class Configuration:
    logger: "Logger"
    simulation: "SimulationConfiguration"
    cpg: "CPGConfiguration"
    random: "RandomConfiguration"
    genetic: "GeneticConfiguration"
    controller: "ControllerConfiguration"

    def __post_init__(self):
        for f in fields(self):
            getattr(self, f.name).set_configuration(self)
