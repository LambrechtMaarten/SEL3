from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.logger.silent_logger import SilentLogger
from configs.subcontrollers.logger.standard_logger import StandardLogger
from configs.subcontrollers.logger.wandb_logger import WandbLogger


def register():
    SubConfigurationMap.add_configuration(
        Logger,
        StandardLogger(),
    )

    SubConfigurationMap.add_configuration(
        Logger,
        SilentLogger(),
    )

    SubConfigurationMap.add_configuration(
        Logger,
        WandbLogger()
    )
