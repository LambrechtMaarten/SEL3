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

class NNControllerPretrain(BaseNNController):
    def __init__(self):
        super().__init__(1 + 10)

    def pretrain():
        pass

    def act(self, cpg_state, control_input, configuration, env_state):
        STOP_THRESHOLD = 0.05 
        
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        # Target reached
        if distance < STOP_THRESHOLD:
            cpg_generator = configuration.cpg.cpg_generator

            return cpg_generator.modulate_body(
                cpg_state,
                cpg_generator.body_to_jarr(
                    cpg_generator.generate(configuration).reset()
                ),
            )
        cpg_generator = configuration.cpg.cpg_generator
        

        angle = jnp.arctan2(deltas[1], deltas[0])
        
        print(f"Going towards: {jnp.degrees(angle)}°")
        print("POSITION: ", robot_pos)

        rot = obs["disk_rotation"][2]

        x = self.build_obs_angle(env_state, angle)

        dist, value = self.model.apply(self.params, x)

        action = dist.mode()
        leading_arm_index = self.angle_to_arm_relative(angle, rot)
        full_body = self.network_output_to_body(action, cpg_state, leading_arm_index)
        return cpg_generator.modulate_body(cpg_state, full_body)