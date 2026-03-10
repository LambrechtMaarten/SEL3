from abc import ABC


class SubConfiguration(ABC):
    """
    This class defines the interface for all sub-configurations used in configuration class.
    The subconfigurations are exchangable configurations that are logged to ensure repeatability.
    """

    def __init__(self, name):
        self.name = name
        self._configuration = None

    def set_configuration(self, config):
        self._configuration = config
