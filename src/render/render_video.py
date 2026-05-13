import os

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import (
    ControllerConfiguration,
)
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.register import register
from configs.subcontrollers.simulation.simulation_configurations import (
    SimulationConfiguration,
)
from src.controller.control_input import ControlInput
from src.environment.environment import Environment
import jax.numpy as jnp


def render_saved_controller(controller_path: str):
    register()
    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "standard"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "crossover"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "network_pretrain"),
    )

    configuration.logger.init_logger()

    controller = configuration.controller.controller
    controller.read_controller(os.path.join(controller_path, "controller_300"))

    env = Environment(configuration)

    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)

    env_state = env.reset(configuration.random.split())
    cpg_state = cpg.reset()

    frames = []
    i = 0
    j = 0
    control_inputs = [
        [2.0,0.0], [-2.0,-2.0], [0.0,0.0]
    ]
    reward = 0.0
    prev_pos = jnp.array([0.0,0.0])
    while i < len(control_inputs) and j < 2000:
        env_state = env_state
        actions = controller.act(
            cpg_state, control_inputs[i], configuration, env_state
        )

        env_state = env.step(actions, env_state)
        STOP_THRESHOLD = 0.075
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        delta = robot_pos - prev_pos
        prev_pos = robot_pos

        direction = jnp.array([jnp.cos(0.0), jnp.sin(0.0)])
        f_v = jnp.dot(delta, direction)

        reward += f_v

        deltas = jnp.array(control_inputs[i]) - robot_pos
        distance = jnp.linalg.norm(deltas)
        if distance < STOP_THRESHOLD:
            i += 1
            j = 0
        j += 1

        frames.append(env.render(env_state))
    print(reward)

    configuration.logger.log_video(frames, "video.mp4")