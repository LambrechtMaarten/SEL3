from configs.subconfiguration import SubConfigurationMap
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.logger.silent_logger import SilentLogger
from configs.subcontrollers.logger.standard_logger import StandardLogger


def register():
    SubConfigurationMap.add_configuration(
        Logger,
        StandardLogger()
    )

    SubConfigurationMap.add_configuration(
        Logger,
        SilentLogger()
    )
