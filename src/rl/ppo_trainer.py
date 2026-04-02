from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from configs.config import Configuration
from src.environment.environment import Environment
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


# ---------- Observaties ----------

def extract_cpg_state_vec(cpg_state: CPGState) -> jarr:
    # Simpel: gebruik de outputs als state
    return cpg_state.outputs.ravel()


def angle_to_vec(theta: float) -> jarr:
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def make_obs(theta: float, cpg_state: CPGState) -> jarr:
    dir_vec = angle_to_vec(theta)
    cpg_vec = extract_cpg_state_vec(cpg_state)
    return jnp.concatenate([dir_vec, cpg_vec], axis=-1)


# ---------- Netwerken ----------

class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs: jarr) -> jarr:
        x = nn.Dense(128)(obs)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs: jarr) -> jarr:
        x = nn.Dense(128)(obs)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x[..., 0]


@dataclass
class Transition:
    obs: jarr
    action: jarr
    logprob: jarr
    value: jarr
    reward: jarr
    done: jarr


# ---------- PPO helpers ----------

def gaussian_logprob(mean: jarr, log_std: jarr, action: jarr) -> jarr:
    std = jnp.exp(log_std)
    var = std ** 2
    return -0.5 * jnp.sum(
        ((action - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages = advantages.at[t].set(gae)
    returns = advantages + values[:-1]
    return advantages, returns


# ---------- Rollout ----------

def rollout_batch(
    rng: jarr,
    configuration: Configuration,
    policy_params: Dict[str, Any],
    value_params: Dict[str, Any],
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    num_episodes: int,
    episode_length: int,
    log_std: jarr,
) -> Tuple[jarr, Dict[str, Any]]:
    env = Environment(configuration)
    cpg_gen = configuration.cpg.cpg_generator

    transitions = []

    for _ in range(num_episodes):
        rng, key_theta, key_env = jax.random.split(rng, 3)
        theta = jax.random.uniform(key_theta, (), minval=0.0, maxval=2 * jnp.pi)

        env_state = env.reset(key_env)
        cpg = cpg_gen.generate(configuration)
        cpg_state = cpg.reset()

        start_x = env_state.observations["disk_position"][0]
        start_y = env_state.observations["disk_position"][1]

        for _ in range(episode_length):
            obs = make_obs(theta, cpg_state)

            mean_action = policy_net.apply(policy_params, obs)
            rng, key_action = jax.random.split(rng)
            std = jnp.exp(log_std)
            action = mean_action + std * jax.random.normal(key_action, mean_action.shape)
            logprob = gaussian_logprob(mean_action, log_std, action)

            # CPG + env
            cpg_state = cpg_gen.modulate_body(cpg_state, action)
            cpg_state = cpg.step(cpg_state)
            actions_env = cpg_gen.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions_env, env_state)

            dx = env_state.observations["disk_position"][0] - start_x
            dy = env_state.observations["disk_position"][1] - start_y
            reward = dx * jnp.cos(theta) + dy * jnp.sin(theta)

            value = value_net.apply(value_params, obs)

            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    logprob=logprob,
                    value=value,
                    reward=reward,
                    done=0.0,
                )
            )

    batch = {
        "obs": jnp.stack([t.obs for t in transitions]),
        "action": jnp.stack([t.action for t in transitions]),
        "logprob": jnp.stack([t.logprob for t in transitions]),
        "value": jnp.stack([t.value for t in transitions]),
        "reward": jnp.stack([t.reward for t in transitions]),
        "done": jnp.stack([t.done for t in transitions]),
    }

    return rng, batch


# ---------- PPO training ----------

def run_ppo_training(configuration: Configuration):
    logger = configuration.logger
    logger.init_logger()

    rng = jax.random.PRNGKey(0)

    # Init CPG om dimensies te kennen
    cpg_gen = configuration.cpg.cpg_generator
    cpg = cpg_gen.generate(configuration)
    cpg_state = cpg.reset()
    dummy_obs = make_obs(0.0, cpg_state)
    obs_dim = dummy_obs.size
    action_dim = cpg_state.outputs.ravel().size

    policy_net = PolicyNetwork(action_dim=action_dim)
    value_net = ValueNetwork()

    rng, key_policy, key_value = jax.random.split(rng, 3)
    policy_params = policy_net.init(key_policy, dummy_obs)
    value_params = value_net.init(key_value, dummy_obs)

    policy_opt = optax.adam(3e-4)
    value_opt = optax.adam(1e-3)
    policy_opt_state = policy_opt.init(policy_params)
    value_opt_state = value_opt.init(value_params)

    log_std = jnp.zeros(action_dim)

    NUM_UPDATES = 50
    EPISODES_PER_UPDATE = 8
    EPISODE_LENGTH = 200
    PPO_EPOCHS = 4
    CLIP_EPS = 0.2
    GAMMA = 0.99
    LAMBDA = 0.95

    for update in range(NUM_UPDATES):
        rng, batch = rollout_batch(
            rng,
            configuration,
            policy_params,
            value_params,
            policy_net,
            value_net,
            EPISODES_PER_UPDATE,
            EPISODE_LENGTH,
            log_std,
        )

        rewards = batch["reward"]
        values = batch["value"]
        dones = batch["done"]

        last_value = values[-1]
        values_ext = jnp.concatenate([values, last_value[None]], axis=0)

        advantages, returns = compute_gae(
            rewards, values_ext, dones, gamma=GAMMA, lam=LAMBDA
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def policy_loss_fn(params, log_std, batch, advantages, old_logprob):
            mean = policy_net.apply(params, batch["obs"])
            logprob = gaussian_logprob(mean, log_std, batch["action"])
            ratio = jnp.exp(logprob - old_logprob)
            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
            loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            return loss

        def value_loss_fn(params, batch, returns):
            preds = value_net.apply(params, batch["obs"])
            return jnp.mean((preds - returns) ** 2)

        old_logprob = batch["logprob"]

        for _ in range(PPO_EPOCHS):
            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(
                policy_params, log_std, batch, advantages, old_logprob
            )
            updates, policy_opt_state = policy_opt.update(
                policy_grads, policy_opt_state
            )
            policy_params = optax.apply_updates(policy_params, updates)

            value_loss, value_grads = jax.value_and_grad(value_loss_fn)(
                value_params, batch, returns
            )
            v_updates, value_opt_state = value_opt.update(
                value_grads, value_opt_state
            )
            value_params = optax.apply_updates(value_params, v_updates)

        ep_returns = rewards.reshape(EPISODES_PER_UPDATE, EPISODE_LENGTH).sum(axis=1)
        avg_return = float(jnp.mean(ep_returns))

        logger.log(
            {
                "update": update,
                "avg_episode_return": avg_return,
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
            }
        )

    # Hier kun je policy_params eventueel opslaan als je dat wilt
