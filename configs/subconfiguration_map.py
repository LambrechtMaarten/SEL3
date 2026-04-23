import json
from typing import Dict, TypeVar

from configs.subconfiguration import SubConfiguration
from configs.subconfigurations.controller.controller_configurations import ControllerConfiguration
from configs.subconfigurations.cpg.cpg_configurations import CPGConfiguration
from configs.subconfigurations.genetic.genetic_configurations import GeneticConfiguration
from configs.subconfigurations.random.random_configurations import RandomConfiguration
from configs.subconfigurations.simulation.simulation_configurations import SimulationConfiguration

T = TypeVar("T", bound=SubConfiguration)


class SubConfigurationMap:
    """
    This singleton class is used to register and look up configurations by name.
    """

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

    @staticmethod
    def get_configuration_from_json(json_str: str, subconfiguration_class: type[T]) -> T:
        json_obj = json.loads(json_str)
        logname_map = {
            SimulationConfiguration: "simulation",
            CPGConfiguration: "cpg",
            RandomConfiguration: "random",
            GeneticConfiguration: "genetic",
            ControllerConfiguration: "controller"
        }
        return SubConfigurationMap.subconfiguration_map[
            subconfiguration_class][json_obj[logname_map[subconfiguration_class]]
        ]
