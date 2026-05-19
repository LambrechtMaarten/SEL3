import os.path

import cv2
import numpy as np

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subconfigurations.controller.controller_configurations import (
    ControllerConfiguration,
)
from configs.subconfigurations.cpg.cpg_configurations import CPGConfiguration
from configs.subconfigurations.genetic.genetic_configurations import GeneticConfiguration
from configs.subconfigurations.logger.logger import Logger
from configs.subconfigurations.random.random_configurations import RandomConfiguration
from configs.subconfigurations.simulation.simulation_configurations import (
    SimulationConfiguration,
)
from src.controller.control_input import ControlInput
from src.environment.environment import Environment


def simulate_controller(output_path: str):
    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "silent_logger"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "evosax"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "network_pretrain"),
    )
    env = Environment(configuration)
    configuration.controller.controller.read_controller(
        os.path.join(output_path, "controller")
    )
    env_state = env.reset(configuration.random.split())

    while True:
        key = -1
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 255:
                break
            key = k

        if key == 27:
            break

        control_input = {
            ord("z"): ControlInput.UP,
            ord("s"): ControlInput.DOWN,
            ord("q"): ControlInput.LEFT,
            ord("d"): ControlInput.RIGHT,
        }.get(key, ControlInput.WAIT)

        # act geeft directe joint actions terug — geen CPG tussenstap
        actions = configuration.controller.controller.act(
            None, control_input, configuration, env_state
        )
        env_state = env.step(actions, env_state)
        frame = env.render(env_state)

        cv2.imshow("Simulation", np.array(frame))

    cv2.destroyAllWindows()
