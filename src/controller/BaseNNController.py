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
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)

        # policy
        mean = nn.Dense(self.action_dim, bias_init=nn.initializers.ones)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)

        dist = distrax.MultivariateNormalDiag(mean, std)

        # value
        value = nn.Dense(1)(x)

        return dist, jnp.squeeze(value, axis=-1)

class BaseNNController(Controller, ABC):
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.params = None
        self.model = ActorCritic(action_dim=self.action_dim)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4)
        )
        self.logger = None
        self.epochs = 6

    @abstractmethod
    def act(self, cpg_state, control_input, configuration, env_state):
        pass

    @staticmethod
    def evaluator(configuration, rng):
        pass

    def angle_reward(self, prev_pos, curr_pos, prev_rot, curr_rot, angle):
        delta = curr_pos[:2] - prev_pos[:2]
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])

        rotation_penalty = jnp.abs(curr_rot - prev_rot)

        return jnp.dot(delta, direction) # - 0.02 *rotation_penalty

    def build_obs_angle(self, env_state, angle, speed=1.0):
        obs = env_state.observations
        # Encodeer hoek als sin/cos zodat 0° en 360° hetzelfde zijn
        angle_enc = jnp.array([jnp.sin(angle), jnp.cos(angle)])
        return jnp.concatenate(
            [
                jnp.array([obs["disk_rotation"][2]]),
                angle_enc,
                jnp.atleast_1d(jnp.asarray(speed, dtype=jnp.float32)),  # doelsnelheid (1D)
                obs["joint_position"],   # huidige gewrichtshoeken (30D)
                obs["joint_velocity"],   # gewrichtssnelheden voor fase-informatie (30D)
            ]
        )

    def network_output_to_body(self, action, cpg_state, leading_arm_idx):
        """Vul freq + amplitudes in vanuit netwerk, hou offsets en phase biases vast."""
        frequency = action[0:1]
        amplitudes = action[1:]

        default_cpg_state = BasicCPGGenerator.modulate_cpg(
            cpg_state=cpg_state,
            leading_arm_index=leading_arm_idx,
            max_joint_limit=1.0,
        )

        return jnp.concatenate(
            [
                frequency,
                amplitudes,
                default_cpg_state.offset_goals,  # vast houden op huidige waarden
                default_cpg_state.coupled_phase_biases.ravel(),  # vast houden
            ]
        )
    
    def get_angles(self, key=None):
        return jnp.arange(5) * (2 * jnp.pi / 5)

    def angle_to_arm_relative(self, angle, disk_rotation_z):
        """Bereken welke arm het dichtst bij de gewenste richting ligt,
        rekening houdend met de huidige rotatie van de robot."""
        # Relatieve hoek: gewenste richting min huidige rotatie
        relative_angle = angle - disk_rotation_z

        arm_angles = self.get_angles()
        diff = jnp.abs((relative_angle - arm_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return jnp.argmin(diff)
    
    def train_controller(self, configuration, num_steps=1000):
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

        def rollout_many(rng, params, angles, speeds):
            keys = jax.random.split(rng, len(angles))
            return jax.vmap(rollout_fn, in_axes=(0, None, 0, 0))(keys, params, angles, speeds)

        for iteration in range(100):
            print(f"Starting iteration {iteration}")
            rng, subkey, angle_key = jax.random.split(rng, 3)
            arm_angles = self.get_angles(angle_key)
            speeds = jnp.ones(len(arm_angles))  # PPO traint bij volle snelheid

            traj = rollout_many(subkey, params, arm_angles, speeds)

            all_obs, all_act, all_logp, all_val, all_rew = traj

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

            print(f"Total reward: {jnp.sum(rew_buf)}")
            print(f"Avg_reward: {jnp.sum(rew_buf) / len(arm_angles)}")
            print(f"Max reward: {jnp.max(jnp.sum(all_rew, axis=1)):.4f}")
            print(f"Min reward: {jnp.min(jnp.sum(all_rew, axis=1)):.4f}")
            print("Average reward per step:", jnp.sum(rew_buf) / len(rew_buf))
            print("Average advantage:", jnp.mean(advantages))
            print("Average return:", jnp.mean(returns))

            batch = (
                jnp.array(obs_buf),
                jnp.array(act_buf),
                jnp.array(logp_buf),
                returns,
                advantages,
            )
            log_data = {
                "total_reward": jnp.sum(rew_buf),
                "average_reward": jnp.sum(rew_buf) / len(arm_angles),
                "average_reward_per_step": jnp.sum(rew_buf) / len(rew_buf),
                "average_advantage": jnp.mean(advantages),
                "average_return": jnp.mean(returns),
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
        print(f"Iter {iteration}, loss {loss}")
        return params, opt_state
    
    def _make_rollout_fn(self, env, model, configuration, num_steps):
        def rollout_fn(rng, params, angle, speed):

            def scan_step(carry, _):
                env_state, rng = carry
                rng, subkey = jax.random.split(rng)

                x = self.build_obs_angle(env_state, angle - env_state.observations["disk_rotation"][2], speed)
                dist, value = model.apply(params, x)

                action = dist.sample(seed=subkey)
                log_prob = dist.log_prob(action)

                prev_pos = env_state.observations["disk_position"]
                prev_rot = env_state.observations["disk_rotation"][2]

                # Netwerk output zijn directe joint actions — geen CPG tussenstap
                env_state = env.step(action, env_state)

                curr_pos = env_state.observations["disk_position"]
                curr_rot = env_state.observations["disk_rotation"][2]
                reward = self.angle_reward(prev_pos, curr_pos, prev_rot, curr_rot, angle)

                return (env_state, rng), (x, action, log_prob, value, reward)

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

    def save_controller(self, logger):
        path = Path(logger.base_folder) / "controller"
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