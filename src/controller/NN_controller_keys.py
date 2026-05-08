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

CONTROL_INPUT_TO_ANGLE = {
    ControlInput.RIGHT: 0.0,
    ControlInput.UP: jnp.pi * 0.4,
    ControlInput.LEFT: jnp.pi * 0.8,
    ControlInput.DOWN: jnp.pi * 1.2,
}

class NNControllerKeys(BaseNNController):
    def __init__(self):
        super().__init__(30)  # 5 armen × 3 segmenten × 2 assen

    def act(self, cpg_state, control_input, configuration, env_state):
        if control_input < ControlInput.WAIT:
            return jnp.zeros(30)

        obs = env_state.observations
        angle = CONTROL_INPUT_TO_ANGLE[control_input]
        rot = obs["disk_rotation"][2]

        x = self.build_obs_angle(env_state, angle - rot, speed=1.0)
        dist, _ = self.model.apply(self.params, x)

        # Directe joint actions — geen CPG tussenstap
        return dist.mode()