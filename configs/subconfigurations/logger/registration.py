from configs.subconfiguration_map import SubConfigurationMap
from configs.subconfigurations.logger.logger import Logger
from configs.subconfigurations.logger.silent_logger import SilentLogger
from configs.subconfigurations.logger.standard_logger import StandardLogger
from configs.subconfigurations.logger.wandb_logger import WandbLogger


def register():
    SubConfigurationMap.add_configuration(
        Logger,
        StandardLogger(),
    )

    SubConfigurationMap.add_configuration(
        Logger,
        SilentLogger(),
    )

    SubConfigurationMap.add_configuration(Logger, WandbLogger())
