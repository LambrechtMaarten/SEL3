from configs.subconfiguration import SubConfiguration
from src.cpg.cpg_generators import CPGGenerator, BasicCPGGenerator, FullyConnecyedCPGGenerator


class CPGConfiguration(SubConfiguration):
    def __init__(
            self,
            name: str,
            cpg_generator: CPGGenerator,
    ):
        super().__init__(name)
        self.cpg_generator: CPGGenerator = cpg_generator
        self.children.append(cpg_generator)


standard = CPGConfiguration(
    "standard",
    BasicCPGGenerator()
)

fully_connected = CPGConfiguration(
    "fully_connected",
    FullyConnecyedCPGGenerator()
)
