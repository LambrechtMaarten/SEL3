from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.random.random_configurations import RandomConfiguration


def register():
    SubConfigurationMap.add_configuration(
        RandomConfiguration,
        RandomConfiguration(
            "standard",
            0
        )
    )
