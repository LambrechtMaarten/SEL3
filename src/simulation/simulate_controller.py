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
        SubConfigurationMap.get_configuration(ControllerConfiguration, "network"),
    )
    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    # Initialiseer het netwerk voor read_controller
    configuration.controller.controller._init_network(configuration)
    configuration.controller.controller.read_controller(
        os.path.join(output_path, "controller")
    )
    cpg = cpg_generator.generate(configuration)
    env_state = env.reset(configuration.random.split())
    cpg_state = cpg.reset()

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
        }.get(key, ControlInput.ZZZ)

        cpg_state = configuration.controller.controller.act(
            cpg_state, control_input, configuration, env_state
        )
        cpg_state = cpg.step(cpg_state)
        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(actions, env_state)
        frame = env.render(env_state)

        cv2.imshow("Simulation", np.array(frame))

    cv2.destroyAllWindows()
