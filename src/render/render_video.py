"""Script to render and save a video of a pre-trained controller navigating waypoints."""

import os

import jax.numpy as jnp

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subconfigurations.controller.controller_configurations import (
    ControllerConfiguration,
)
from configs.subconfigurations.cpg.cpg_configurations import CPGConfiguration
from configs.subconfigurations.genetic.genetic_configurations import GeneticConfiguration
from configs.subconfigurations.logger.logger import Logger
from configs.subconfigurations.random.random_configurations import RandomConfiguration
from configs.subconfigurations.register import register
from configs.subconfigurations.simulation.simulation_configurations import (
    SimulationConfiguration,
)
from src.environment.environment import Environment


def render_saved_controller(controller_path: str):
    """Load a saved controller and render a multi-waypoint navigation episode.

    Constructs a standard configuration, loads the controller from disk, then
    runs the simulation until the robot reaches each of three hardcoded
    waypoints (or a maximum of 1000 steps per waypoint).  The rendered frames
    are logged as a video.

    Args:
        controller_path: Directory containing the saved ``controller`` file.
    """
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
    controller.read_controller(os.path.join(controller_path, "controller"))

    env = Environment(configuration)

    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)

    env_state = env.reset(configuration.random.split())
    cpg_state = cpg.reset()

    frames = []
    i = 0
    j = 0
    control_inputs = [[2.0, 0.0], [-2.0, -2.0], [0.0, 0.0]]
    while i < len(control_inputs) and j < 1000:
        env_state = env_state
        actions = controller.act(cpg_state, control_inputs[i], configuration, env_state)

        env_state = env.step(actions, env_state)
        STOP_THRESHOLD = 0.075
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]

        deltas = jnp.array(control_inputs[i]) - robot_pos
        distance = jnp.linalg.norm(deltas)
        if distance < STOP_THRESHOLD:
            i += 1
            j = 0
        j += 1

        frames.append(env.render(env_state))

    configuration.logger.log_video(frames, "video.mp4")
