import cv2
import numpy as np

from configs.config import Configuration
from src.controller.controller import Input
from src.environment.environment import Environment


def simulate_controller(configuration: Configuration):
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
