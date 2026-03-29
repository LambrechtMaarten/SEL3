from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from src.controller.direction_controller import DirectionController
from src.controller.one_direction_controller import OneDirectionController
from src.controller.symmetry_controller import BasicSymmetryController


def register():
    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("standard", OneDirectionController())
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("body_direction", DirectionController())
    )

