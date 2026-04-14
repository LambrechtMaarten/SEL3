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
        freq_amp = nn.Dense(1 + 10, bias_init=nn.initializers.ones)(x)  # hoge bias
        offsets = nn.Dense(10, bias_init=nn.initializers.zeros)(x)  # nul bias
        mean = jnp.concatenate([freq_amp, offsets], axis=-1)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)

        dist = distrax.MultivariateNormalDiag(mean, std)

        # value
        value = nn.Dense(1)(x)

        return dist, jnp.squeeze(value, axis=-1)


def network_output_to_body(action, cpg_state):
    """Vul freq + amplitudes in vanuit netwerk, hou offsets en phase biases vast."""
    frequency = action[0:1]
    n = cpg_state.amplitude_goals.size  # 10
    amplitudes = action[1 : 1 + n]
    offsets = action[1 + n :]

    default_cpg_state = BasicCPGGenerator.modulate_cpg(
        cpg_state=cpg_state,
        leading_arm_index=0,
        max_joint_limit=1.0,
    )

    return jnp.concatenate(
        [
            frequency,
            amplitudes,
            offsets,  # vast houden op huidige waarden
            default_cpg_state.coupled_phase_biases.ravel(),  # vast houden
        ]
    )


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
        full_body = network_output_to_body(action, cpg_state)
        return cpg_generator.modulate_body(cpg_state, full_body)

    def train_controller(self, configuration, num_steps=800, epochs=8):
        logger = configuration.logger
        logger.init_logger()
        rng = configuration.random.rng
        env = Environment(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        cpg_reset = cpg.reset()
        action_dim = 1 + cpg_reset.amplitude_goals.size + cpg_reset.offset_goals.size
        model = ActorCritic(action_dim=action_dim)

        dummy_env = env.reset(rng)
        dummy_input = build_obs(dummy_env, ControlInput.RIGHT)
        params = model.init(rng, dummy_input)
        optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
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

                    prev_x = env_state.observations["disk_position"][0]
                    full_body = network_output_to_body(action, cpg_state)
                    cpg_state = cpg_generator.modulate_body(cpg_state, full_body)
                    cpg_state = cpg.step(cpg_state)

                    env_state = env.step(
                        cpg_generator.outputs_to_actions(
                            cpg_state.outputs, configuration
                        ),
                        env_state,
                    )

                    reward = env_state.observations["disk_position"][0] - prev_x

                    return (cpg_state, env_state, rng), (
                        x,
                        action,
                        log_prob,
                        value,
                        reward,
                    )

                init_cpg_state = BasicCPGGenerator.modulate_cpg(
                    cpg_state=cpg_generator.generate(configuration).reset(),
                    leading_arm_index=0,
                    max_joint_limit=1.0,
                )

                init_cpg_state_full = cpg_generator.modulate_body(
                    cpg_generator.generate(configuration).reset(),
                    jnp.zeros(
                        cpg_generator.body_to_jarr(
                            cpg_generator.generate(configuration).reset()
                        ).size
                    ),
                )

                init = (
                    init_cpg_state,
                    env.reset(rng),
                    rng,
                )

                (_, _, _), traj = jax.lax.scan(scan_step, init, None, length=num_steps)

                return traj

            return jax.jit(rollout_fn)

        rollout_fn = make_rollout_fn(
            env, cpg_generator, model, configuration, num_steps
        )
        for iteration in range(100):
            rng, subkey = jax.random.split(rng)
            print(f"Starting iteration {iteration}")
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            val_buf = []

            traj = rollout_fn(subkey, params)

            obs_buf, act_buf, logp_buf, val_buf, rew_buf = traj

            # bootstrap value
            _, last_val = model.apply(params, obs_buf[-1])
            val_buf = jnp.append(val_buf, last_val)

            # log rewards
            print(f"Total reward: {sum(rew_buf)}")
            print(f"Max reward per step: {jnp.max(rew_buf):.4f}")
            print(f"Min reward per step: {jnp.min(rew_buf):.4f}")
            print(f"Final x position: {obs_buf[-1][0]:.4f}")
            print("Average reward per step:", sum(rew_buf) / len(rew_buf))

            advantages = compute_gae(rew_buf, val_buf, jnp.zeros(len(rew_buf)))

            returns = advantages + jnp.array(val_buf[:-1])
            print("Average advantage:", jnp.mean(advantages))
            print("Average return:", jnp.mean(returns))
            advantages = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + 1e-8
            )  # then normalize

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

            logger.log(
                {
                    "total_reward": sum(rew_buf),
                    "average_reward": sum(rew_buf) / len(rew_buf),
                    "average_advantage": jnp.mean(advantages),
                    "stdev_advantage": jnp.std(advantages),
                    "average_return": jnp.mean(returns),
                    "loss": loss,
                },
            )
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
