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
        SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
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
    control_inputs = [
        [2.0,2.0], [2.0,0.0], [-2.0,-2.0], [0.0,0.0]
    ]

    while i < len(control_inputs) and j < 2500:
        env_state = env_state
        cpg_state = controller.act(
            cpg_state, control_inputs[i], configuration, env_state
        )

        cpg_state = cpg.step(cpg_state)

        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)

        env_state = env.step(actions, env_state)
        STOP_THRESHOLD = 0.05 
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
