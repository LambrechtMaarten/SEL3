from abc import ABC, abstractmethod
import pickle
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_generators.basic_cpg_generator import BasicCPGGenerator
from src.environment.environment import Environment
from src.controller.NN_controller_pretrain import NNControllerPretrain

class NNControllerAngle(NNControllerPretrain):
    def __init__(self):
        super().__init__()

    def act(self, cpg_state, control_input, configuration, env_state):
        STOP_THRESHOLD = 0.05

        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        angle = control_input[0]
        speed = control_input[1]

        # Doel bereikt: geen beweging
        if distance < STOP_THRESHOLD or speed == 0.0:
            return jnp.zeros(30)
        
        # Snelheid: proportioneel aan afstand tot doel, verzadigt op 1.0
        # (robot vertraagt automatisch als het doel nadert)
        speed = np.clip(speed, 0.001, 1.0)

        rot = obs["disk_rotation"][2]

        relative_angle = angle - rot
        local_angle, sector = self.to_local_angle_and_sector(relative_angle)

        rng = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
        
        print("SPEED: ", speed)
        print("SECTOR: ", sector)
        print("ANGLE: ", (local_angle / jnp.pi)*180)
        robot_pos = obs["disk_position"][0:2]
        print("POS: ", robot_pos)

        x = self.build_obs_angle(env_state, local_angle, sector, speed)

        dist, _ = self.model.apply(self.params, x)
        
        actions = dist.mode()
        shift = 6 * sector
        rotated_actions = jnp.roll(actions, shift)

        # Directe joint actions — geen CPG tussenstap
        return rotated_actions