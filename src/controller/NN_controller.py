import pickle
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from src.controller.control_input import ControlInput
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
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)

        # policy
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)

        dist = distrax.MultivariateNormalDiag(mean, std)

        # value
        value = nn.Dense(1)(x)

        return dist, jnp.squeeze(value, axis=-1)


class NNController(Controller):
    def __init__(self):
        self.params = None
        self.model = None

    def act(self, cpg_state, control_input, configuration, env_state):
        if control_input == ControlInput.WAIT:
            cpg_generator = configuration.cpg.cpg_generator
            return cpg_generator.modulate_body(
                cpg_state,
                cpg_generator.body_to_jarr(
                    cpg_generator.generate(configuration).reset()
                ),
            )
        cpg_generator = configuration.cpg.cpg_generator

        obs = env_state.observations

        x = jnp.concatenate(
            [
                obs["disk_position"],
                obs["disk_rotation"],
                jax.nn.one_hot(control_input.value, 5),
            ]
        )

        dist, value = self.model.apply(self.params, x)

        action = dist.mode()

        return cpg_generator.modulate_body(cpg_state, action)

    def train_controller(self, configuration, num_steps=2048, epochs=10):
        logger = configuration.logger
        logger.init_logger()
        rng = configuration.random.rng
        env = Environment(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        action_dim = cpg_generator.body_to_jarr(cpg.reset()).size
        model = ActorCritic(action_dim=action_dim)

        dummy_env = env.reset(rng)
        dummy_input = build_obs(dummy_env, ControlInput.RIGHT)
        params = model.init(rng, dummy_input)
        optimizer = optax.adam(3e-4)
        opt_state = optimizer.init(params)

        def make_rollout_fn(env, cpg_generator, model, configuration, num_steps):
            def rollout_fn(rng, params):
                def scan_step(carry, _):
                    cpg_state, env_state, rng = carry

                    rng, subkey = jax.random.split(rng)

                    x = build_obs(env_state, ControlInput.RIGHT)

                    dist, value = model.apply(params, x)
                    action = dist.sample(seed=subkey)
                    log_prob = dist.log_prob(action)

                    cpg_state = cpg_generator.modulate_body(cpg_state, action)
                    cpg_state = cpg.step(cpg_state)

                    env_state = env.step(
                        cpg_generator.outputs_to_actions(
                            cpg_state.outputs, configuration
                        ),
                        env_state,
                    )

                    reward = env_state.observations["disk_position"][0]

                    return (cpg_state, env_state, rng), (
                        x,
                        action,
                        log_prob,
                        value,
                        reward,
                    )

                init = (
                    cpg_generator.modulate_body(
                        cpg_generator.generate(configuration).reset(),
                        jnp.zeros(
                            cpg_generator.body_to_jarr(
                                cpg_generator.generate(configuration).reset()
                            ).size
                        ),
                    ),
                    env.reset(rng),
                    rng,
                )

                (_, _, _), traj = jax.lax.scan(scan_step, init, None, length=num_steps)

                return traj

            return jax.jit(rollout_fn)

        rollout_fn = make_rollout_fn(
            env, cpg_generator, model, configuration, num_steps
        )
        for iteration in range(1000):
            print(f"Starting iteration {iteration}")
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            val_buf = []

            traj = rollout_fn(rng, params)

            obs_buf, act_buf, logp_buf, val_buf, rew_buf = traj

            # bootstrap value
            _, last_val = model.apply(params, obs_buf[-1])
            val_buf = jnp.append(val_buf, last_val)

            # log rewards
            print(f"Total reward: {sum(rew_buf)}")
            print("Average reward per step:", sum(rew_buf) / len(rew_buf))
            logger.log(
                {
                    "total_reward": sum(rew_buf),
                    "average_reward": sum(rew_buf) / len(rew_buf),
                },
            )

            # Normaliseer advantages
            advantages = compute_gae(rew_buf, val_buf, jnp.zeros(len(rew_buf)))
            mean = jnp.mean(advantages)
            std = jnp.std(advantages)
            advantages = (advantages - mean) / (std + 1e-8)

            returns = advantages + jnp.array(val_buf[:-1])
            print("Average advantage:", jnp.mean(advantages))
            print("Average return:", jnp.mean(returns))
            logger.log(
                {
                    "average_advantage": jnp.mean(advantages),
                    "stdev_advantage": jnp.std(advantages),
                    "average_return": jnp.mean(returns),
                },
            )

            batch = (
                jnp.array(obs_buf),
                jnp.array(act_buf),
                jnp.array(logp_buf),
                returns,
                advantages,
            )

            for _ in range(epochs):
                params, opt_state, loss = update_step_jit(
                    params, opt_state, model, batch, optimizer
                )

            logger.log({"loss": loss})
            print(f"Iter {iteration}, loss {loss}")
        # Na afloop van de training, sla de parameters op
        self.params = params
        self.model = model
        self.save_controller(configuration.logger)

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

    @staticmethod
    def evaluator(configuration, rng):
        return super().evaluator(configuration, rng)

    def genome_size(self, configuration):
        flat_params, _ = jax.flatten_util.ravel_pytree(self.params)
        return flat_params.shape[0]


def build_obs(env_state, control_input):
    obs = env_state.observations

    return jnp.concatenate(
        [
            obs["disk_position"],
            obs["disk_rotation"],
            jax.nn.one_hot(control_input.value, 5),
        ]
    )


def step_env(env, cpg, cpg_generator, params, model, rng, state, configuration):
    cpg_state, env_state, control_input = state

    x = build_obs(env_state, control_input)

    dist, value = model.apply(params, x)
    rng, subkey = jax.random.split(rng)
    action = dist.sample(seed=subkey)
    log_prob = dist.log_prob(action)

    # 👉 NN output → CPG body
    cpg_state = cpg_generator.modulate_body(cpg_state, action)
    cpg_state = cpg.step(cpg_state)

    env_state = env.step(
        cpg_generator.outputs_to_actions(cpg_state.outputs, configuration), env_state
    )

    reward = env_state.observations["disk_position"][0]

    return (cpg_state, env_state, control_input), (x, action, log_prob, value, reward)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return jnp.array(advantages)


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
