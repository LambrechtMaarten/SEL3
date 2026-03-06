from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.cpg_configurations import CPGConfiguration
    from configs.simulation_configurations import SimulationConfiguration
    from configs.random_configurations import RandomConfiguration


class Configuration:
    def __init__(self,
                 simulation_configuration: "SimulationConfiguration",
                 cpg_configuration: "CPGConfiguration",
                 random_configuration: "RandomConfiguration"):
        self.simulation: "SimulationConfiguration" = simulation_configuration
        self.cpg: "CPGConfiguration" = cpg_configuration
        self.random: "RandomConfiguration" = random_configuration
        for sub_config in [self.simulation, self.cpg]:
            sub_config.set_configuration(self)
