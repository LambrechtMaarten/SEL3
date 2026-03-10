from configs.subconfiguration import SubConfiguration
from src.cpg.cpg_generators import CPGGenerator, BasicCPGGenerator, FullyConnectedCPGGenerator


class CPGConfiguration(SubConfiguration):
    """
    This class contains the configuration of the CPG.
    It determines among other things how many oscilators there are per arm and how these map to actuators.
    """

    def __init__(
            self,
            name: str,
            cpg_generator: CPGGenerator):
        super().__init__(name)
        self.cpg_generator: CPGGenerator = cpg_generator


standard = lambda: CPGConfiguration(
    "standard",
    BasicCPGGenerator()
)

fully_connected = lambda: CPGConfiguration(
    "fully_connected",
    FullyConnectedCPGGenerator()
)
