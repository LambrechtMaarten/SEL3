from abc import ABC

from configs.config import Configuration


class SubConfiguration(ABC):
    """
    This class defines the interface for all sub-configurations used in configuration class.
    The subconfigurations are exchangable configurations that are logged to ensure repeatability.
    """

    def __init__(self, name):
        self.name = name
        self._configuration = None

    def set_configuration(self, configuration: Configuration) -> None:
        """
        This method sets the configuration for this subconfiguration.
        It is needed because the subconfigurations are arguments of the configuration class and so can't get it in the constructor.
        :param configuration: the configuration to be set
        """
        self._configuration = configuration
