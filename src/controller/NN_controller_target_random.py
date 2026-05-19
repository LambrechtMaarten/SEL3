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

class NNControllerTargetRandom(BaseNNController):
    def __init__(self):
        super().__init__(1 + 10 + 10)

    def _make_rollout_fn(self, env, cpg_generator, model, configuration, num_steps):
        def rollout_fn(rng, params, angle):

            def scan_step(carry, _):
                cpg_state, env_state, rng = carry
                rng, subkey = jax.random.split(rng)

                x = self.build_obs_angle(env_state, angle)
                dist, value = model.apply(params, x)

                action = dist.sample(seed=subkey)
                log_prob = dist.log_prob(action)

                prev_pos = env_state.observations["disk_position"]
                prev_rot = env_state.observations["disk_rotation"][2]

                full_body = self.network_output_to_body(
                    action, cpg_state
                )

                cpg_state = cpg_generator.modulate_body(cpg_state, full_body)
                cpg_state = cpg_generator.generate(configuration).step(cpg_state)

                env_state = env.step(
                    cpg_generator.outputs_to_actions(
                        cpg_state.outputs, configuration
                    ),
                    env_state,
                )

                curr_pos = env_state.observations["disk_position"]
                curr_rot = env_state.observations["disk_rotation"][2]
                reward = self.angle_reward(prev_pos, curr_pos, prev_rot, curr_rot, angle)

                return (cpg_state, env_state, rng), (
                    x, action, log_prob, value, reward
                )

            init = (
                cpg_generator.generate(configuration).reset(),
                env.reset(rng),
                rng,
            )

            (_, _, _, _), traj = jax.lax.scan(scan_step, init, None, length=num_steps)
            return traj

        return jax.jit(rollout_fn)

    def network_output_to_body(self, action, cpg_state):
        """Vul freq + amplitudes in vanuit netwerk, hou offsets en phase biases vast."""
        frequency = action[0:1]
        amplitudes = action[1:11]
        offsets = action[11:]

        return jnp.concatenate(
            [
                frequency,
                amplitudes,
                offsets,
                self.phases, # vast houden
            ]
        )
    
    def get_angles(self, key):
        return jax.random.uniform(key, minval=0.0, maxval=2 * jnp.pi)

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
        full_body = self.network_output_to_body(action, cpg_state)
        return cpg_generator.modulate_body(cpg_state, full_body)