import jax.numpy as jnp
import numpy as np

from src.controller.NN_controller_pretrain import NNControllerPretrain
from src.controller.control_input import ControlInput

class NNControllerAngle(NNControllerPretrain):
    def __init__(self):
        super().__init__()

    def act(self, cpg_state, control_input: ControlInput, configuration, env_state):
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        angle = control_input.angle
        speed = control_input.speed

        # Doel bereikt: geen beweging
        if distance < self.stop_threshold or speed == 0.0:
            return jnp.zeros(30)
        
        # Snelheid: clip zodat snelheid altijd tussen 0 en 1 blijft
        speed = np.clip(speed, 0.001, 1.0)

        rot = obs["disk_rotation"][2]

        relative_angle = angle - rot
        local_angle, sector = self.to_local_angle_and_sector(relative_angle)

        x = self.build_obs_angle(obs, local_angle, sector, speed)

        dist, _ = self.model.apply(self.params, x)
        
        actions = dist.mode()
        shift = 6 * sector
        rotated_actions = jnp.roll(actions, shift)

        # Directe joint actions — geen CPG tussenstap
        return rotated_actions