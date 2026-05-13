from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from src.cpg.cpg_generators.basic_cpg_generator import BasicCPGGenerator
from src.cpg.cpg_generators.fully_connected_cpg_generator import (
    FullyConnectedCPGGenerator,
)
from src.cpg.cpg_generators.fully_connected_symmetric_cpg_generator import FullyConnectedSymmetricCPGGenerator


def register():
    SubConfigurationMap.add_configuration(
        CPGConfiguration, CPGConfiguration("standard", BasicCPGGenerator())
    )

    SubConfigurationMap.add_configuration(
        CPGConfiguration,
        CPGConfiguration("fully_connected", FullyConnectedCPGGenerator()),
    )

    SubConfigurationMap.add_configuration(
        CPGConfiguration,
        CPGConfiguration("symmetric", FullyConnectedSymmetricCPGGenerator()),
    )
