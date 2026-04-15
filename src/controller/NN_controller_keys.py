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
        super().__init__()

    def act(self, cpg_state, control_input, configuration, env_state):        
        if control_input < ControlInput.WAIT:
            cpg_generator = configuration.cpg.cpg_generator

            return cpg_generator.modulate_body(
                cpg_state,
                cpg_generator.body_to_jarr(
                    cpg_generator.generate(configuration).reset()
                ),
            )
        cpg_generator = configuration.cpg.cpg_generator

        obs = env_state.observations
        angle = CONTROL_INPUT_TO_ANGLE[control_input]
        rot = obs["disk_rotation"][2]

        x = self.build_obs_angle(env_state, angle)

        dist, value = self.model.apply(self.params, x)

        action = dist.mode()
        leading_arm_index = self.angle_to_arm_relative(angle, rot)
        full_body = self.network_output_to_body(action, cpg_state, leading_arm_index)
        return cpg_generator.modulate_body(cpg_state, full_body)