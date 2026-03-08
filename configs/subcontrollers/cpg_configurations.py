from configs.subconfiguration import SubConfiguration
from src.cpg.cpg_generators import CPGGenerator, BasicCPGGenerator, FullyConnectedCPGGenerator


class CPGConfiguration(SubConfiguration):
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
