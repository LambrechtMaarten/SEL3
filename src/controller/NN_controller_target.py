from abc import ABC, abstractmethod
import pickle
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_generators.basic_cpg_generator import BasicCPGGenerator
from src.environment.environment import Environment
from src.controller.BaseNNController import BaseNNController

class NNControllerTarget(BaseNNController):
    def __init__(self):
        super().__init__(30)  # 5 armen × 3 segmenten × 2 assen

    def act(self, cpg_state, control_input, configuration, env_state):
        STOP_THRESHOLD = 0.05

        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        # Doel bereikt: geen beweging
        if distance < STOP_THRESHOLD:
            return jnp.zeros(30)

        def wrap_angle(angle):
            return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

        angle = wrap_angle(jnp.arctan2(deltas[1], deltas[0]))
        rot = obs["disk_rotation"][2]

        # Snelheid proportioneel aan afstand: robot vertraagt automatisch bij het doel
        speed = jnp.clip(distance, 0.0, 1.0)

        x = self.build_obs_angle(env_state, angle - rot, speed)
        dist, _ = self.model.apply(self.params, x)

        # Directe joint actions — geen CPG tussenstap
        return dist.mode()