from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import (
    ControllerConfiguration,
)
from src.controller.NN_controller import NNController
from src.controller.NN_controller_target import NNControllerTarget
from src.controller.one_direction_controller import OneDirectionController


def register():
    SubConfigurationMap.add_configuration(
        ControllerConfiguration, ControllerConfiguration("network", NNController())
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("network_multi", NNControllerTarget()),
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("standard", OneDirectionController()),
    )
