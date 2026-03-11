from abc import ABC
from typing import TypeVar, Dict

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


T = TypeVar("T", bound=SubConfiguration)


class SubConfigurationMap:
    subconfiguration_map: Dict[type[SubConfiguration], Dict[str, SubConfiguration]] = dict()

    @staticmethod
    def add_configuration(subconfiguration_class: type[T], subconfiguration: T):
        if subconfiguration_class not in SubConfigurationMap.subconfiguration_map:
            SubConfigurationMap.subconfiguration_map[subconfiguration_class] = dict()

        if subconfiguration.name in SubConfigurationMap.subconfiguration_map[subconfiguration_class]:
            raise Exception(f'Subconfiguration "{subconfiguration.name}" '
                            f'already defined for {subconfiguration_class.__name__}')

        SubConfigurationMap.subconfiguration_map[subconfiguration_class][subconfiguration.name] = subconfiguration

    @staticmethod
    def get_configuration(subconstructor_class: type[T], name) -> T:
        return SubConfigurationMap.subconfiguration_map[subconstructor_class][name]
