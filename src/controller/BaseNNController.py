"""Base neural-network controller implementing shared PPO training infrastructure."""

import pickle
from pathlib import Path
from typing import Any, Callable, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from configs.subconfigurations.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr


def ppo_loss(
    params: Any,
    model: nn.Module,
    batch: tuple[jarr, jarr, jarr, jarr, jarr],
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> jarr:
    """Compute the clipped PPO loss with value and entropy terms.

    Args:
        params: Current Flax model parameters.
        model: The ActorCritic Flax module.
        batch: Tuple of (observations, actions, old_log_probs, returns,
            advantages), each a JAX array.
        clip_eps: PPO clipping epsilon for the policy ratio.
        vf_coef: Coefficient for the value function loss term.
        ent_coef: Coefficient for the entropy bonus term.

    Returns:
        Scalar loss value combining policy loss, value loss, and entropy.
    """
    obs, actions, old_log_probs, returns, advantages = batch

    dist, values = model.apply(params, obs)
    log_probs = dist.log_prob(actions)

    ratio = jnp.exp(log_probs - old_log_probs)

    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    value_loss = jnp.mean((returns - values) ** 2)

    entropy = jnp.mean(dist.entropy())

    return policy_loss + vf_coef * value_loss - ent_coef * entropy


def update_step(
    params: Any,
    opt_state: optax.OptState,
    model: nn.Module,
    batch: tuple[jarr, jarr, jarr, jarr, jarr],
    optimizer: optax.GradientTransformation,
) -> tuple[Any, optax.OptState, jarr]:
    """Perform a single gradient update step using the PPO loss.

    Args:
        params: Current Flax model parameters.
        opt_state: Current Optax optimizer state.
        model: The ActorCritic Flax module.
        batch: Tuple of (observations, actions, old_log_probs, returns,
            advantages).
        optimizer: Optax gradient transformation to apply.

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value).
    """
    loss, grads = jax.value_and_grad(ppo_loss)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


update_step_jit = jax.jit(update_step, static_argnames=["model", "optimizer"])


class ActorCritic(nn.Module):
    """Flax module implementing a shared-encoder Actor-Critic network.

    The shared encoder produces a 256-dimensional hidden representation that
    is reused by both the policy head and the value head, reducing the total
    parameter count and promoting feature sharing.

    Attributes:
        action_dim: Dimensionality of the continuous action space (30 for the
            brittle star with 5 arms × 3 segments × 2 joints).
    """

    action_dim: int

    @nn.compact
    def __call__(self, x):
        """Forward pass through the Actor-Critic network.

        Args:
            x: Observation vector of shape ``(obs_dim,)``.

        Returns:
            Tuple of (dist, value) where ``dist`` is a
            ``distrax.MultivariateNormalDiag`` action distribution and
            ``value`` is a scalar state-value estimate.
        """
        # shared encoder
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)

        # policy
        mean = nn.Dense(self.action_dim, bias_init=nn.initializers.ones)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)

        dist = distrax.MultivariateNormalDiag(mean, std)

        # value
        value = nn.Dense(1)(x)

        return dist, jnp.squeeze(value, axis=-1)


class BaseNNController(Controller):
    """Base class for neural-network controllers trained with PPO.

    Provides the ActorCritic model, the PPO training loop, observation
    construction, reward computation, GAE advantage estimation, and
    checkpoint utilities.  Concrete subclasses extend this class to add
    pre-training phases (e.g. Behavioral Cloning) or specialised action
    interfaces.

    Attributes:
        stop_threshold: Distance threshold below which the robot stops moving
            toward the current target.
        action_dim: Number of joint actions (5 arms × 3 segments × 2 joints
            = 30).
        params: Current Flax model parameter tree, or None before
            initialisation.
        model: ActorCritic Flax module.
        optimizer: Optax gradient transformation (gradient clipping + Adam).
        logger: Logger instance, set during training.
        epochs: Number of PPO update epochs per iteration.
        speeds: List of absolute target speeds gathered during pre-training.
        norm_speeds: List of normalised target speeds in [0, 1].
    """

    def __init__(self):
        """Initialise the BaseNNController with default hyperparameters."""
        self.stop_threshold = 0.05
        self.action_dim = 5 * 3 * 2  # 5 arms, 3 segments, 2 joints
        self.params = None
        self.model = ActorCritic(action_dim=self.action_dim)
        self.optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
        self.logger = None
        self.epochs = 6
        self.speeds = []
        self.norm_speeds = []

    @staticmethod
    def evaluator(configuration: Configuration, rng):
        """Not implemented for NN controllers; raises an exception.

        Args:
            configuration: Global simulation and training configuration.
            rng: JAX random key (unused).

        Raises:
            Exception: Always, because NN controllers are not optimised via
                the genetic evaluator interface.
        """
        raise Exception("Not implemented")

    def angle_reward(
        self,
        prev_pos: jarr,
        curr_pos: jarr,
        angle: float,
        speed_target: float,
    ) -> tuple[jarr, jarr, jarr]:
        """Compute the directional locomotion reward for one simulation step.

        The reward encourages the robot to move in the desired direction at
        the target speed.  It combines a forward-progress term with a
        speed-tracking penalty.

        Args:
            prev_pos: Robot position at the previous step, shape ``(3,)``.
            curr_pos: Robot position at the current step, shape ``(3,)``.
            angle: Target heading angle in world frame (radians).
            speed_target: Target forward speed in simulation units per step.

        Returns:
            Tuple of (total_reward, forward_reward, speed_reward) as scalar
            JAX arrays.
        """
        delta = curr_pos[:2] - prev_pos[:2]
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])

        forward_velocity = jnp.dot(delta, direction)

        speed_error = (forward_velocity - speed_target) ** 2

        speed_reward = -speed_error * 10

        return (forward_velocity * 0.3) + speed_reward, forward_velocity * 0.3, speed_reward

    def build_obs_angle(
        self,
        obs: dict[str, jarr],
        angle: float,
        sector: int = 0,
        speed: float = 1.0,
    ) -> jarr:
        """Construct the observation vector fed to the neural network.

        The observation encodes the target heading and speed together with
        arm-relative joint positions, velocities, and contact signals.
        Observations are rolled so that the leading arm always appears first,
        exploiting the 5-fold rotational symmetry of the robot.

        Args:
            obs: Dictionary of environment observations keyed by name.
            angle: Target heading in the local (arm-relative) frame (radians).
            sector: Index of the arm sector chosen as the leading arm (0–4).
            speed: Normalised target speed in [0, 1].

        Returns:
            Concatenated observation vector of shape ``(93,)`` composed of
            2D angle encoding (sin/cos), 1D speed, 30D joint positions,
            30D joint velocities, and 30D segment contacts.
        """
        # Encode angle as sin/cos so that 0° and 360° map to the same representation
        angle_enc = jnp.array([jnp.sin(angle), jnp.cos(angle)])

        return jnp.concatenate(
            [
                angle_enc,
                jnp.atleast_1d(jnp.asarray(speed, dtype=jnp.float32)),  # target speed (1D)
                jnp.roll(obs["joint_position"], 6 * sector),  # current joint angles (30D)
                jnp.roll(
                    obs["joint_velocity"], 6 * sector
                ),  # joint velocities for phase information (30D)
                jnp.roll(obs["segment_contact"], 6 * sector),  # segment contact signals (30D)
            ]
        )

    def to_local_angle_and_sector(self, relative_angle: float) -> Tuple[float, int]:
        """Convert a world-frame relative angle to a sector index and local residual angle.

        The brittle star has 5-fold rotational symmetry.  This method
        identifies which of the 5 arm sectors the target heading falls into
        and computes the small residual angle within that sector.

        Args:
            relative_angle: Target angle expressed relative to the robot's
                current heading (radians).

        Returns:
            Tuple of (local_angle, sector_index) where ``local_angle`` is the
            residual angle within the sector (in radians) and ``sector_index``
            is an integer in [0, 4] identifying the nearest arm.
        """
        angle = jnp.mod(relative_angle + jnp.pi, 2 * jnp.pi) - jnp.pi
        sector_size = 2 * jnp.pi / 5
        k_raw = jnp.round(angle / sector_size).astype(int)
        k_idx = k_raw % 5
        local_angle = angle - k_raw * sector_size

        return (local_angle, k_idx)

    def get_speeds(self):
        """Return the number of speed levels and the maximum speed for training.

        If pre-training has populated ``self.speeds``, the values collected
        during Behavioral Cloning are used.  Otherwise sensible defaults are
        returned.

        Returns:
            Tuple of (n, max_speed) where ``n`` is the number of parallel
            rollouts to run and ``max_speed`` is the maximum target speed in
            simulation units per step.
        """
        n = 15
        max_speed = 0.0036
        if len(self.speeds) > 0:
            n = len(self.speeds)
            max_speed = self.speeds[-1]
        return n, max_speed

    def train_controller(self, configuration, num_steps=500):
        """Train the controller using Proximal Policy Optimisation (PPO).

        Runs 1000 PPO iterations.  Each iteration collects parallel rollouts
        across randomly sampled heading angles and speeds, computes GAE
        advantages, and performs ``self.epochs`` gradient update steps.
        Checkpoints are saved every 100 iterations.

        Args:
            configuration: Global simulation and training configuration.
            num_steps: Number of simulation steps per rollout trajectory.
        """
        if self.logger is None:
            self.logger = configuration.logger
            self.logger.init_logger()
        rng = configuration.random.rng

        env = Environment(configuration)

        dummy_env = env.reset(rng)
        dummy_input = self.build_obs_angle(dummy_env.observations, 0.0)
        if self.params is None:
            params = self.model.init(rng, dummy_input)
            self.params = params
        else:
            params = self.params  # use pre-trained params as starting point

        opt_state = self.optimizer.init(params)

        rollout_fn = self._make_rollout_fn(env, self.model, configuration, num_steps)

        def rollout_many(rng, params, angles, speeds, norm_speeds):
            keys = jax.random.split(rng, len(angles))
            return jax.vmap(rollout_fn, in_axes=(0, None, 0, 0, 0))(
                keys, params, angles, speeds, norm_speeds
            )

        for iteration in range(1000):
            # Save checkpoint every 100 iterations
            if iteration % 100 == 0 and iteration != 0:
                self.params = params
                self.save_controller(self.logger, f"controller_{iteration}")

            rng, subkey, angle_key, speed_key2 = jax.random.split(rng, 4)

            n, max_speed = self.get_speeds()

            # World-frame target heading: uniformly random over 360°
            arm_angles = jax.random.uniform(angle_key, shape=(n,), minval=0.0, maxval=2 * jnp.pi)

            # Interpolate across known speeds for better generalisation
            norm_speeds_random = jax.random.uniform(speed_key2, shape=(n,), minval=0.01, maxval=1.0)
            speeds_random = norm_speeds_random * max_speed  # real m/s in simulator for reward
            traj = rollout_many(subkey, params, arm_angles, speeds_random, norm_speeds_random)

            (
                all_obs,
                all_act,
                all_logp,
                all_val,
                all_rew,
                all_forward_rewards,
                all_speed_penalties,
            ) = traj

            obs_buf = all_obs.reshape(-1, all_obs.shape[-1])
            act_buf = all_act.reshape(-1, all_act.shape[-1])
            logp_buf = all_logp.reshape(-1)
            rew_buf = all_rew.reshape(-1)

            _, last_vals = jax.vmap(lambda o: self.model.apply(params, o[-1]))(all_obs)
            val_bufs = jax.vmap(lambda v, lv: jnp.append(v, lv))(all_val, last_vals)

            advantages_list = [
                self.compute_gae(r, v, jnp.zeros(len(r))) for r, v in zip(all_rew, val_bufs)
            ]
            returns_list = [adv + v[:-1] for adv, v in zip(advantages_list, val_bufs)]

            advantages = jnp.concatenate(
                [(adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8) for adv in advantages_list]
            )
            returns = jnp.concatenate(returns_list)

            batch = (
                jnp.array(obs_buf),
                jnp.array(act_buf),
                jnp.array(logp_buf),
                returns,
                advantages,
            )

            total_forward_reward = jnp.sum(all_forward_rewards)
            total_speed_penalty = jnp.sum(all_speed_penalties)

            log_data = {
                "total_reward": jnp.sum(rew_buf),
                "average_reward": jnp.sum(rew_buf) / len(arm_angles),
                "average_reward_per_step": jnp.sum(rew_buf) / len(rew_buf),
                "average_return": jnp.mean(returns),
                "avg_forward_reward_per_step": total_forward_reward / len(rew_buf),
                "avg_speed_reward_per_step": total_speed_penalty / len(rew_buf),
                "average_advantage": jnp.mean(advantages),
            }

            params, opt_state = self.update_and_log(params, opt_state, batch, iteration, log_data)

        self.params = params
        self.save_controller(configuration.logger)

    def update_and_log(
        self,
        params: Any,
        opt_state: optax.OptState,
        batch: tuple[jarr, jarr, jarr, jarr, jarr],
        iteration: int,
        log_data: dict[str, Any],
    ) -> tuple[Any, optax.OptState]:
        """Run multiple PPO update epochs, log metrics, and return updated state.

        Args:
            params: Current Flax model parameters.
            opt_state: Current Optax optimizer state.
            batch: Tuple of (observations, actions, old_log_probs, returns,
                advantages).
            iteration: Current training iteration index (used for logging).
            log_data: Dictionary of scalar metrics to be logged alongside the
                loss value.

        Returns:
            Tuple of (updated_params, updated_opt_state).
        """
        for _ in range(self.epochs):
            params, opt_state, loss = update_step_jit(
                params, opt_state, self.model, batch, self.optimizer
            )
        log_data["loss"] = loss
        self.logger.log(log_data)
        return params, opt_state

    def _make_rollout_fn(
        self,
        env: Environment,
        model: nn.Module,
        configuration: Configuration,
        num_steps: int,
    ) -> Callable:
        """Build a JIT-compiled rollout function for a single episode.

        The returned function runs a fixed-length episode in the MuJoCo
        environment.  At each step the network selects an action given the
        current observation, the environment is stepped, and a reward is
        computed from the directional progress.

        Args:
            env: Wrapped MuJoCo environment.
            model: ActorCritic Flax module.
            configuration: Global simulation and training configuration.
            num_steps: Number of simulation steps per rollout.

        Returns:
            JIT-compiled callable with signature
            ``(rng, params, angle, speed, norm_speed) -> trajectory``.
        """

        def rollout_fn(rng, params, angle, speed, norm_speed):
            def scan_step(carry, _):
                env_state, rng = carry
                rng, subkey = jax.random.split(rng)
                relative_angle = angle - env_state.observations["disk_rotation"][2]

                local_angle, k_idx = self.to_local_angle_and_sector(relative_angle)

                x = self.build_obs_angle(env_state.observations, local_angle, k_idx, norm_speed)
                dist, value = model.apply(params, x)

                action = dist.sample(seed=subkey)
                action_world = jnp.roll(action, 6 * k_idx)

                log_prob = dist.log_prob(action)

                prev_pos = env_state.observations["disk_position"]

                # Network outputs are direct joint actions — no CPG intermediate step
                env_state = env.step(action_world, env_state)

                curr_pos = env_state.observations["disk_position"]
                reward, forward_reward, speed_reward = self.angle_reward(
                    prev_pos, curr_pos, angle, speed
                )

                return (env_state, rng), (
                    x,
                    action,
                    log_prob,
                    value,
                    reward,
                    forward_reward,
                    speed_reward,
                )

            init = (env.reset(rng), rng)

            (_, _), traj = jax.lax.scan(scan_step, init, None, length=num_steps)
            return traj

        return jax.jit(rollout_fn)

    def compute_gae(
        self,
        rewards: jarr,
        values: jarr,
        dones: jarr,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> jarr:
        """Compute Generalised Advantage Estimation (GAE) for a trajectory.

        Args:
            rewards: Reward sequence of shape ``(T,)``.
            values: Value estimates of shape ``(T+1,)`` where the last element
                is the bootstrap value for the terminal state.
            dones: Binary done-flags of shape ``(T,)``.
            gamma: Discount factor.
            lam: GAE lambda parameter controlling the bias-variance trade-off.

        Returns:
            Advantage estimates of shape ``(T,)``.
        """
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return jnp.array(advantages)

    def save_controller(self, logger: Logger, name: str = "controller"):
        """Serialise the current model parameters to disk using pickle.

        Args:
            logger: Logger instance whose ``base_folder`` is used as the
                output directory.
            name: Filename stem for the saved parameter file.
        """
        path = Path(logger.base_folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def read_controller(self, path: str):
        """Load model parameters from a pickle file and reconstruct the model.

        The action dimension is inferred from the saved parameter shapes so
        that the correct ActorCritic architecture is restored automatically.

        Args:
            path: Filesystem path to the pickle file containing Flax params.
        """
        with open(path, "rb") as f:
            self.params = pickle.load(f)
            self.model = ActorCritic(action_dim=self.params["params"]["Dense_2"]["kernel"].shape[1])

    def genome_size(self, configuration: Configuration):
        """Return the total number of scalar parameters in the flattened model.

        Args:
            configuration: Global simulation and training configuration
                (unused, present for interface compatibility).

        Returns:
            Integer number of scalar parameters in the network.
        """
        flat_params, _ = jax.flatten_util.ravel_pytree(self.params)
        return flat_params.shape[0]

    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
        env_state: BaseEnvState,
    ):
        """Compute joint actions from the current environment state.

        Converts the world-frame target into a sector-relative angle,
        constructs the observation vector, queries the policy in deterministic
        mode (mode of the distribution), and rotates the actions back to the
        world frame.

        Args:
            cpg_state: Unused; present for interface compatibility.
            control_input: High-level locomotion command carrying the target
                position as a 2-element array.
            configuration: Global simulation and training configuration.
            env_state: Current MuJoCo environment state.

        Returns:
            JAX array of shape ``(30,)`` containing joint torques, or zeros
            if the target has been reached.
        """
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        # Target reached: no movement
        if distance < self.STOP_THRESHOLD:
            return jnp.zeros(30)

        angle = jnp.arctan2(deltas[1], deltas[0])
        rot = obs["disk_rotation"][2]

        relative_angle = angle - rot
        local_angle, sector = self.to_local_angle_and_sector(relative_angle)

        # Speed: proportional to distance to target, saturates at 1.0
        # (robot automatically slows down as it approaches the goal)
        speed = np.clip(distance, 0.001, 1.0)

        x = self.build_obs_angle(obs, local_angle, sector, speed)

        dist, _ = self.model.apply(self.params, x)

        actions = dist.mode()
        shift = 6 * sector
        rotated_actions = jnp.roll(actions, shift)

        # Direct joint actions — no CPG intermediate step
        return rotated_actions
