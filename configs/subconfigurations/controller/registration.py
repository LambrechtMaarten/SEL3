from configs.subconfiguration_map import SubConfigurationMap
from configs.subconfigurations.controller.controller_configurations import (
    ControllerConfiguration,
)
from src.controller.BaseNNController import BaseNNController
from src.controller.NN_controller_angle import NNControllerAngle
from src.controller.NN_controller_pretrain import NNControllerPretrain
from src.controller.one_direction_map_elites_controller import OneDirectionMapElitesController


def register():
    SubConfigurationMap.add_configuration(
        ControllerConfiguration,
        ControllerConfiguration("network", BaseNNController()),
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
        ControllerConfiguration("map elites", OneDirectionMapElitesController()),
    )
