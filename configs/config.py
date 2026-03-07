from typing import TYPE_CHECKING
from dataclasses import dataclass, fields

if TYPE_CHECKING:
    from configs.logger import Logger
    from configs.cpg_configurations import CPGConfiguration
    from configs.random_configurations import RandomConfiguration
    from configs.genetic_configurations import GeneticConfiguration
    from configs.controller_configurations import ControllerConfiguration
    from configs.simulation_configurations import SimulationConfiguration


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
