"""NN controller that accepts a direct angle and speed command via ControlInput."""

import jax.numpy as jnp
import numpy as np
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from src.controller.control_input import ControlInput
from src.controller.NN_controller_pretrain import NNControllerPretrain
from src.cpg.cpg_state import CPGState


class NNControllerAngle(NNControllerPretrain):
    """Neural-network controller driven by an explicit angle and speed command.

    Extends :class:`NNControllerPretrain` with an ``act`` implementation that
    reads the desired heading angle and speed directly from the
    :class:`ControlInput` rather than deriving them from a target position.
    All training logic is inherited from the parent class.
    """

    def __init__(self):
        """Initialise NNControllerAngle (delegates to parent)."""
        super().__init__()

    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
        env_state: BaseEnvState,
    ):
        """Compute joint actions given an angle/speed locomotion command.

        If the robot is already within ``stop_threshold`` of the notional
        target, or the commanded speed is zero, all joints are held at rest.
        Otherwise the target heading is converted to a sector-relative angle,
        the observation is constructed, and the policy is queried in
        deterministic mode.

        Args:
            cpg_state: Unused; present for interface compatibility.
            control_input: Locomotion command containing ``angle`` (world
                frame, radians) and ``speed`` (normalised, 0–1).
            configuration: Global simulation and training configuration.
            env_state: Current MuJoCo environment state.

        Returns:
            JAX array of shape ``(30,)`` containing joint torques.
        """
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        angle = control_input.angle
        speed = control_input.speed

        # Target reached or zero speed commanded: no movement
        if distance < self.stop_threshold or speed == 0.0:
            return jnp.zeros(30)

        # Clip speed to always stay between 0 and 1
        speed = np.clip(speed, 0.001, 1.0)

        rot = obs["disk_rotation"][2]

        relative_angle = angle - rot
        local_angle, sector = self.to_local_angle_and_sector(relative_angle)

        x = self.build_obs_angle(obs, local_angle, sector, speed)

        dist, _ = self.model.apply(self.params, x)

        actions = dist.mode()
        shift = 6 * sector
        rotated_actions = jnp.roll(actions, shift)

        # Direct joint actions — no CPG intermediate step
        return rotated_actions
