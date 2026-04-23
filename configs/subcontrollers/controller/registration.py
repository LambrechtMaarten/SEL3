from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import (
    ControllerConfiguration,
)
from src.controller.network_controller import NetworkController
from src.controller.one_direction_controller import OneDirectionController
from src.controller.one_direction_map_elites_controller import OneDirectionMapElitesController


def register():
    SubConfigurationMap.add_configuration(
        ControllerConfiguration, ControllerConfiguration("network", NetworkController())
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("standard", OneDirectionController()),
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("map elites", OneDirectionMapElitesController()),
    )
