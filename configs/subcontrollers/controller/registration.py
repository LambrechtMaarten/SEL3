from configs.subconfiguration import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from src.controller import StandardController, BodyDirectionController


def register():
    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("standard", StandardController())
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("body_direction", BodyDirectionController())
    )
