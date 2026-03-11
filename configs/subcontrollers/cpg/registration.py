from configs.subconfiguration import SubConfigurationMap
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from src.cpg.cpg_generators import BasicCPGGenerator, FullyConnectedCPGGenerator


def register():
    SubConfigurationMap.add_configuration(
        CPGConfiguration,
        CPGConfiguration("standard", BasicCPGGenerator())
    )

    SubConfigurationMap.add_configuration(
        CPGConfiguration,
        CPGConfiguration("fully_connected", FullyConnectedCPGGenerator())
    )
