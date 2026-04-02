from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from configs.config import Configuration
from src.environment.environment import Environment
from src.controller.network_controller import NetworkController
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr


@dataclass
class Transition:
    direction: jarr
    state_vec: jarr
    action: jarr
    logprob: jarr
    value: jarr
    reward: jarr
    done: jarr


class ValueNetwork(nn.Module):
    state_dim: int

    @nn.compact
    def __call__(self, direction_vec: jarr, state_vec: jarr) -> jarr:
        x = jnp.concatenate([direction_vec, state_vec], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x[..., 0]


def extract_state_from_cpg(cpg_state: CPGState) -> jarr:
    return cpg_state.outputs.ravel()


def angle_to_vector(theta: float) -> jarr:
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


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


def rollout_batch(
    rng: jarr,
    configuration: Configuration,
    controller: NetworkController,
    controller_params: Dict[str, Any],
    value_params: Dict[str, Any],
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
        direction_vec = angle_to_vector(theta)

        env_state = env.reset(key_env)
        cpg = cpg_gen.generate(configuration)
        cpg_state = cpg.reset()

        start_x = env_state.observations["disk_position"][0]
        start_y = env_state.observations["disk_position"][1]

        for _ in range(episode_length):
            state_vec = extract_state_from_cpg(cpg_state)

            mean_action = controller.policy_apply(
                controller_params, float(theta), cpg_state
            )

            rng, key_action = jax.random.split(rng)
            std = jnp.exp(log_std)
            action = mean_action + std * jax.random.normal(key_action, mean_action.shape)
            logprob = gaussian_logprob(mean_action, log_std, action)

            cpg_state = cpg_gen.modulate_body(cpg_state, action)
            cpg_state = cpg.step(cpg_state)
            actions_env = cpg_gen.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions_env, env_state)

            dx = env_state.observations["disk_position"][0] - start_x
            dy = env_state.observations["disk_position"][1] - start_y
            reward = dx * jnp.cos(theta) + dy * jnp.sin(theta)

            value = value_net.apply(value_params, direction_vec, state_vec)

            transitions.append(
                Transition(
                    direction=direction_vec,
                    state_vec=state_vec,
                    action=action,
                    logprob=logprob,
                    value=value,
                    reward=reward,
                    done=0.0,
                )
            )

    batch = {
        "direction": jnp.stack([t.direction for t in transitions]),
        "state_vec": jnp.stack([t.state_vec for t in transitions]),
        "action": jnp.stack([t.action for t in transitions]),
        "logprob": jnp.stack([t.logprob for t in transitions]),
        "value": jnp.stack([t.value for t in transitions]),
        "reward": jnp.stack([t.reward for t in transitions]),
        "done": jnp.stack([t.done for t in transitions]),
    }

    return rng, batch


def run_ppo_training(configuration: Configuration):
    logger = configuration.logger
    controller: NetworkController = configuration.controller.controller

    rng = jax.random.PRNGKey(0)

    # Policy params
    rng, key_policy = jax.random.split(rng)
    controller_params = controller.init_params(configuration, key_policy)

    # Value net init
    cpg_gen = configuration.cpg.cpg_generator
    cpg = cpg_gen.generate(configuration)
    cpg_state = cpg.reset()
    state_dim = extract_state_from_cpg(cpg_state).size
    value_net = ValueNetwork(state_dim=state_dim)
    rng, key_value = jax.random.split(rng)
    value_params = value_net.init(
        key_value, angle_to_vector(0.0), jnp.zeros(state_dim)
    )

    # Optimizers
    policy_opt = optax.adam(3e-4)
    value_opt = optax.adam(1e-3)
    policy_opt_state = policy_opt.init(controller_params)
    value_opt_state = value_opt.init(value_params)

    # Fixed log_std
    log_std = jnp.zeros_like(
        controller.policy_apply(controller_params, 0.0, cpg_state)
    )

    NUM_UPDATES = 20
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
            controller,
            controller_params,
            value_params,
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
            # Hier gebruiken we alleen state_vec + direction uit batch
            mean = controller.policy_apply(
                params,
                0.0,  # theta zit impliciet in direction_vec; voor nu laten we dit zo
                cpg_state,  # in een nettere versie zou je hier echte obs meegeven
            )
            logprob = gaussian_logprob(mean, log_std, batch["action"])
            ratio = jnp.exp(logprob - old_logprob)
            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
            loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            return loss

        def value_loss_fn(params, batch, returns):
            preds = value_net.apply(
                params, batch["direction"], batch["state_vec"]
            )
            return jnp.mean((preds - returns) ** 2)

        old_logprob = batch["logprob"]

        for _ in range(PPO_EPOCHS):
            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(
                controller_params, log_std, batch, advantages, old_logprob
            )
            updates, policy_opt_state = policy_opt.update(
                policy_grads, policy_opt_state
            )
            controller_params = optax.apply_updates(controller_params, updates)

            value_loss, value_grads = jax.value_and_grad(value_loss_fn)(
                value_params, batch, returns
            )
            v_updates, value_opt_state = value_opt.update(
                value_grads, value_opt_state
            )
            value_params = optax.apply_updates(value_params, v_updates)

        # Gemiddelde return per episode
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

    # Optioneel: policy opslaan in controller.weights
    controller.weights = jnp.array(
        [float(x) for x in controller.flatten_params(controller_params)]
    )
