import pickle
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np

from src.controller.controller import Controller
from src.environment.environment import Environment

def update_step(params, opt_state, model, batch, optimizer):
    loss, grads = jax.value_and_grad(ppo_loss)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


update_step_jit = jax.jit(update_step, static_argnames=["model", "optimizer"])

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
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
    def __init__(self, action_dim):
        self.stop_threshold = 0.05
        self.action_dim = action_dim
        self.params = None
        self.model = ActorCritic(action_dim=self.action_dim)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4)
        )
        self.logger = None
        self.epochs = 6
        self.speeds = []
        self.norm_speeds = []

    @staticmethod
    def evaluator(configuration, rng):
        pass

    def angle_reward(self, prev_pos, curr_pos, angle, speed_target):
        delta = curr_pos[:2] - prev_pos[:2]
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])

        forward_velocity = jnp.dot(delta, direction)

        speed_error = (forward_velocity - speed_target) ** 2

        speed_reward = -speed_error * 10

        return (forward_velocity * 0.3) + speed_reward, forward_velocity*0.3, speed_reward

    def build_obs_angle(self, env_state, angle, sector=0, speed=1.0):
        obs = env_state.observations
        # Encodeer hoek als sin/cos zodat 0° en 360° hetzelfde zijn
        angle_enc = jnp.array([jnp.sin(angle), jnp.cos(angle)])

        return jnp.concatenate(
            [
                angle_enc,
                jnp.atleast_1d(jnp.asarray(speed, dtype=jnp.float32)),  # doelsnelheid (1D)
                jnp.roll(obs["joint_position"], 6 * sector),  # huidige gewrichtshoeken (30D)
                jnp.roll(obs["joint_velocity"], 6 * sector), # gewrichtssnelheden voor fase-informatie (30D)
                jnp.roll(obs["segment_contact"], 6 * sector) # contact van segmenten (30D)
            ]
        )
    
    def to_local_angle_and_sector(self, relative_angle):
        angle = jnp.mod(relative_angle + jnp.pi, 2 * jnp.pi) - jnp.pi
        sector_size = 2 * jnp.pi / 5
        k_raw = jnp.round(angle / sector_size).astype(int)
        k_idx = k_raw % 5
        local_angle = angle - k_raw * sector_size 
        
        return (local_angle, k_idx)
    
    def train_controller(self, configuration, num_steps=500):
        """Train de controller via PPO.

        Args:
            configuration: trainings-configuratie.
            num_steps: stappen per rollout.
            pretrained_body_cpg: optionele platte CPG-array van een expert-gait.
                Als opgegeven, wordt eerst Behavioral Cloning + value warm-up
                uitgevoerd, gevolgd door PPO met KL-regularisatie.
        """
        if self.logger is None:
            self.logger = configuration.logger
            self.logger.init_logger()
        rng = configuration.random.rng

        env = Environment(configuration)

        dummy_env = env.reset(rng)
        dummy_input = self.build_obs_angle(dummy_env, 0.0)
        if self.params is None:
            params = self.model.init(rng, dummy_input)
            self.params = params
        else:
            params = self.params  # gebruik pretrained params als startpunt

        opt_state = self.optimizer.init(params)

        rollout_fn = self._make_rollout_fn(env, self.model, configuration, num_steps)

        def rollout_many(rng, params, angles, speeds, norm_speeds):
            keys = jax.random.split(rng, len(angles))
            return jax.vmap(rollout_fn, in_axes=(0, None, 0, 0, 0))(keys, params, angles, speeds, norm_speeds)

        for iteration in range(1000):
            if iteration % 100 == 0 and iteration != 0:
                self.params = params
                self.save_controller(self.logger, f"controller_{iteration}")
            rng, subkey, speed_key = jax.random.split(rng, 3)

            # Wereldframe doelhoek: volledig random over 360°
            arm_angles = jax.random.uniform(subkey, shape=(len(self.speeds),), minval=0.0, maxval=2*jnp.pi)
            
            # Interpoleer tussen bekende speeds voor betere generalisatie
            max_speed = self.speeds[-1]
            norm_speeds_random = jax.random.uniform(speed_key, shape=(len(self.speeds),), minval=0.01, maxval=1.0)
            speeds_random = norm_speeds_random * max_speed  # echte m/s in simulator voor reward
            traj = rollout_many(subkey, params, arm_angles, speeds_random, norm_speeds_random)  # consistent!

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
                self.compute_gae(r, v, jnp.zeros(len(r)))
                for r, v in zip(all_rew, val_bufs)
            ]
            returns_list = [
                adv + v[:-1]
                for adv, v in zip(advantages_list, val_bufs)
            ]

            advantages = jnp.concatenate([
                (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
                for adv in advantages_list
            ])
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
                "avg_forward_reward_per_step": total_forward_reward/len(rew_buf),
                "avg_speed_reward_per_step": total_speed_penalty/len(rew_buf),
                "average_advantage": jnp.mean(advantages),
            }

            params, opt_state = self.update_and_log(params, opt_state, batch, iteration, log_data)

        self.params = params
        self.save_controller(configuration.logger)

    def update_and_log(self, params, opt_state, batch, iteration, log_data):
        for _ in range(self.epochs):
            params, opt_state, loss = update_step_jit(
                params, opt_state, self.model, batch, self.optimizer
            ) 
        log_data["loss"] = loss
        self.logger.log(log_data)
        return params, opt_state
    
    def _make_rollout_fn(self, env, model, configuration, num_steps):
        def rollout_fn(rng, params, angle, speed, norm_speed):

            def scan_step(carry, _):
                env_state, rng = carry
                rng, subkey = jax.random.split(rng)
                relative_angle = angle - env_state.observations["disk_rotation"][2]
                
                local_angle, k_idx = self.to_local_angle_and_sector(relative_angle)

                x = self.build_obs_angle(env_state, local_angle, k_idx, norm_speed)
                dist, value = model.apply(params, x)

                action = dist.sample(seed=subkey)
                action_world = jnp.roll(action, 6 * k_idx)

                log_prob = dist.log_prob(action)

                prev_pos = env_state.observations["disk_position"]

                # Netwerk output zijn directe joint actions — geen CPG tussenstap
                env_state = env.step(action_world, env_state)

                curr_pos = env_state.observations["disk_position"]
                reward, forward_reward, speed_reward = self.angle_reward(prev_pos, curr_pos, angle, speed)

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

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return jnp.array(advantages)

    def save_controller(self, logger, name="controller"):
        path = Path(logger.base_folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def read_controller(self, path: str):
        with open(path, "rb") as f:
            self.params = pickle.load(f)
            self.model = ActorCritic(
                action_dim=self.params["params"]["Dense_2"]["kernel"].shape[1]
            )

    def genome_size(self, configuration):
        flat_params, _ = jax.flatten_util.ravel_pytree(self.params)
        return flat_params.shape[0]
    
    def act(self, cpg_state, control_input, configuration, env_state):
        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        # Doel bereikt: geen beweging
        if distance < self.STOP_THRESHOLD:
            return jnp.zeros(30)

        angle = jnp.arctan2(deltas[1], deltas[0])
        rot = obs["disk_rotation"][2]

        relative_angle = angle - rot
        local_angle, sector = self.to_local_angle_and_sector(relative_angle)

        # Snelheid: proportioneel aan afstand tot doel, verzadigt op 1.0
        # (robot vertraagt automatisch als het doel nadert)
        speed = np.clip(distance, 0.001, 1.0)

        x = self.build_obs_angle(env_state, local_angle, sector, speed)

        dist, _ = self.model.apply(self.params, x)
        
        actions = dist.mode()
        shift = 6 * sector
        rotated_actions = jnp.roll(actions, shift)

        # Directe joint actions geen CPG tussenstap
        return rotated_actions
    


def ppo_loss(params, model, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    obs, actions, old_log_probs, returns, advantages = batch

    dist, values = model.apply(params, obs)
    log_probs = dist.log_prob(actions)

    ratio = jnp.exp(log_probs - old_log_probs)

    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    value_loss = jnp.mean((returns - values) ** 2)

    entropy = jnp.mean(dist.entropy())

    return policy_loss + vf_coef * value_loss - ent_coef * entropy