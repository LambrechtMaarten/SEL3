from typing import TYPE_CHECKING
from dataclasses import dataclass, fields

if TYPE_CHECKING:
    from configs.logger import Logger
    from configs.cpg_configurations import CPGConfiguration
    from configs.random_configurations import RandomConfiguration
    from configs.genetic_configuration import GeneticConfiguration
    from configs.simulation_configurations import SimulationConfiguration


@dataclass
class Configuration:
    simulation: "SimulationConfiguration"
    cpg: "CPGConfiguration"
    random: "RandomConfiguration"
    logger: "Logger"
    genetic: "GeneticConfiguration"

    def __post_init__(self):
        for f in fields(self):
            getattr(self, f.name).set_configuration(self)
