import cv2
import numpy as np

from configs.config import Configuration
from configs.subconfiguration import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.register import register
from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration
from src.controller import Input
from src.environment import Environment


def main():
    register()
    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "standard"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "fully_connected"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "short"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "body_direction")
    )
    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    configuration.controller.controller.read_controller("../output/2026_03_10-10.36.18/controller")
    cpg = cpg_generator.generate(configuration)
    env_state = env.reset(configuration.random.split())
    cpg_state = cpg.reset()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # escape
            break
        control_input = {
            ord("z"): Input.UP,
            ord("s"): Input.DOWN,
            ord("q"): Input.LEFT,
            ord("d"): Input.RIGHT,
            ord("a"): Input.TURN_LEFT,
            ord("e"): Input.TURN_RIGHT,
        }.get(key, Input.ZZZ)

        cpg_state = configuration.controller.controller.act(cpg_state, control_input, configuration)
        cpg_state = cpg.step(cpg_state)
        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(actions, env_state)
        frame = env.render(env_state)

        cv2.imshow("Simulation", np.array(frame))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
