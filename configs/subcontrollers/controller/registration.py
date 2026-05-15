from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import (
    ControllerConfiguration,
)
from src.controller.NN_controller import NNController
from src.controller.NN_controller_target import NNControllerTarget
from src.controller.NN_controller_pretrain import NNControllerPretrain
from src.controller.one_direction_controller import OneDirectionController
from src.controller.NN_controller_angle import NNControllerAngle


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
        ControllerConfiguration("network_pretrain", NNControllerPretrain()),
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("angle", NNControllerAngle()),
    )

    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("standard", OneDirectionController()),
    )
