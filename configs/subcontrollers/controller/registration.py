from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from src.controller.body_direction_controller import BodyDirectionController
from src.controller.one_direction_body_controller import OneDirectionBodyController
from src.controller.network_controller import NetworkController


def register():
    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("standard", OneDirectionBodyController())
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("body_direction", BodyDirectionController())
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("network", NetworkController())
    )
