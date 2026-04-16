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


# === Behavioral Cloning ===

def bc_loss(params, model, obs_batch, expert_actions):
    """MSE tussen policy-gemiddelde en expert-acties."""
    dist, _ = model.apply(params, obs_batch)
    return jnp.mean((dist.mean() - expert_actions) ** 2)


def bc_update_step(params, opt_state, model, obs_batch, expert_actions, optimizer):
    loss, grads = jax.value_and_grad(bc_loss)(params, model, obs_batch, expert_actions)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


bc_update_jit = jax.jit(bc_update_step, static_argnames=["model", "optimizer"])


# === Value warm-up (enkel value head trainen) ===

def value_warmup_loss(params, model, batch):
    obs, _, _, returns, _ = batch
    _, values = model.apply(params, obs)
    return jnp.mean((returns - values) ** 2)


def value_warmup_step(params, opt_state, model, batch, optimizer):
    loss, grads = jax.value_and_grad(value_warmup_loss)(params, model, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


value_warmup_jit = jax.jit(value_warmup_step, static_argnames=["model", "optimizer"])


# === PPO met KL-regularisatie ===

def ppo_loss_with_kl(params, model, batch, pretrained_params, kl_coef,
                     clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO-loss met KL-divergentieterm om de policy dicht bij de pre-getrainde
    initialisatie te houden. kl_coef neemt af doorheen training (annealing)."""
    obs, actions, old_log_probs, returns, advantages = batch

    dist, values = model.apply(params, obs)
    pretrained_dist, _ = model.apply(pretrained_params, obs)

    log_probs = dist.log_prob(actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    value_loss = jnp.mean((returns - values) ** 2)
    entropy = jnp.mean(dist.entropy())

    # KL(huidige policy || pre-getrainde policy)
    kl = jnp.mean(dist.kl_divergence(pretrained_dist))

    return policy_loss + vf_coef * value_loss - ent_coef * entropy + kl_coef * kl


def update_step_kl(params, opt_state, model, batch, optimizer, pretrained_params, kl_coef):
    loss, grads = jax.value_and_grad(ppo_loss_with_kl)(
        params, model, batch, pretrained_params, kl_coef
    )
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


update_step_kl_jit = jax.jit(update_step_kl, static_argnames=["model", "optimizer"])

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
        mean = nn.Dense(1 + 10, bias_init=nn.initializers.ones)(x)
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
        self.model = None
        self.model = ActorCritic(action_dim=self.action_dim)

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

    def build_obs_angle(self, env_state, angle):
        obs = env_state.observations
        # Encodeer hoek als sin/cos zodat 0° en 360° hetzelfde zijn
        angle_enc = jnp.array([jnp.sin(angle), jnp.cos(angle)])
        return jnp.concatenate(
            [
                # obs["disk_position"][0:2],
                jnp.array([obs["disk_rotation"][2]]),
                angle_enc,
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

    def train_controller(self, configuration, num_steps=1000, epochs=6,
                         pretrained_body_cpg=None):
        """Train de controller via PPO.

        Args:
            configuration: trainings-configuratie.
            num_steps: stappen per rollout.
            epochs: PPO-epochs per iteratie.
            pretrained_body_cpg: optionele platte CPG-array van een expert-gait.
                Als opgegeven, wordt eerst Behavioral Cloning + value warm-up
                uitgevoerd, gevolgd door PPO met KL-regularisatie.
        """
        logger = configuration.logger
        logger.init_logger()
        rng = configuration.random.rng

        env = Environment(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)

        dummy_env = env.reset(rng)
        dummy_input = self.build_obs_angle(dummy_env, 0.0)
        params = self.model.init(rng, dummy_input)
        self.params = params

        # === Pretraining via Behavioral Cloning (optioneel) ===
        pretrained_params = None
        if pretrained_body_cpg is not None:
            self.pretrain_bc(configuration, pretrained_body_cpg)
            pretrained_params = self.params
            params = self.params

        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4)
        )
        opt_state = optimizer.init(params)

        rollout_fn = self._make_rollout_fn(
            env, cpg_generator, self.model, configuration, num_steps
        )

        def rollout_many(rng, params, angles):
            keys = jax.random.split(rng, len(angles))
            return jax.vmap(rollout_fn, in_axes=(0, None, 0))(keys, params, angles)

        # KL-annealing: start bij 0.1, halveert elke ~35 iteraties
        kl_coef_init = jnp.array(0.1)
        kl_decay = jnp.array(50.0)

        for iteration in range(250):
            print(f"Starting iteration {iteration}")
            rng, subkey, angle_key = jax.random.split(rng, 3)
            arm_angles = self.get_angles(angle_key)

            traj = rollout_many(subkey, params, arm_angles)

            all_obs, all_act, all_logp, all_val, all_rew = traj

            obs_buf = all_obs.reshape(-1, all_obs.shape[-1])
            act_buf = all_act.reshape(-1, all_act.shape[-1])
            logp_buf = all_logp.reshape(-1)
            rew_buf = all_rew.reshape(-1)

            _, last_vals = jax.vmap(lambda o: self.model.apply(params, o[-1]))(all_obs)
            val_bufs = jax.vmap(lambda v, lv: jnp.append(v, lv))(all_val, last_vals)

            advantages_list = [
                compute_gae(r, v, jnp.zeros(len(r)))
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

            # Exponentiële afbouw van de KL-coëfficiënt
            kl_coef = kl_coef_init * jnp.exp(-jnp.array(iteration, dtype=jnp.float32) / kl_decay)

            for _ in range(epochs):
                if pretrained_params is not None:
                    params, opt_state, loss = update_step_kl_jit(
                        params, opt_state, self.model, batch, optimizer,
                        pretrained_params, kl_coef
                    )
                else:
                    params, opt_state, loss = update_step_jit(
                        params, opt_state, self.model, batch, optimizer
                    )

            log_data = {
                "total_reward": jnp.sum(rew_buf),
                "average_reward": jnp.sum(rew_buf) / len(arm_angles),
                "average_reward_per_step": jnp.sum(rew_buf) / len(rew_buf),
                "average_advantage": jnp.mean(advantages),
                "average_return": jnp.mean(returns),
                "loss": loss,
            }
            if pretrained_params is not None:
                log_data["kl_coef"] = kl_coef

            logger.log(log_data)
            print(f"Iter {iteration}, loss {loss}")

        self.params = params
        self.save_controller(configuration.logger)

    def _make_expert_rollout_fn(self, env, cpg_generator, cpg, configuration, num_steps):
        """Rollout waarbij de expert-CPG de robot bestuurt.
        Verzamelt observaties zodat we die kunnen gebruiken voor BC."""
        def expert_rollout(rng, expert_cpg_state, angle):
            def scan_step(carry, _):
                cpg_state, env_state = carry
                obs = self.build_obs_angle(env_state, angle)
                cpg_state = cpg.step(cpg_state)
                actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
                env_state = env.step(actions, env_state)
                return (cpg_state, env_state), obs

            init = (expert_cpg_state, env.reset(rng))
            (_, _), obs_seq = jax.lax.scan(scan_step, init, None, length=num_steps)
            return obs_seq  # (num_steps, obs_dim)

        return jax.jit(expert_rollout)

    def pretrain_bc(self, configuration, body_cpg,
                    bc_iterations=100, warmup_iterations=20,
                    num_rollout_steps=500, noise_std=0.02, bc_lr=1e-3):
        """Twee-fasen pretraining vóór PPO.

        Fase 1 – Behavioral Cloning:
          Verzamel observaties door de expert-gait (voor alle 5 armen via symmetrie)
          te spelen. Train de policy head via MSE op de expert-acties.

        Fase 2 – Value warm-up:
          Draai rollouts met de BC-getrainde policy en train enkel de value head
          zodat de advantage-schattingen kloppen bij de start van PPO.

        Args:
            configuration: trainings-configuratie.
            body_cpg: platte JAX-array met de CPG-parameters van de rechts-gait.
            bc_iterations: aantal BC-updateslagen.
            warmup_iterations: aantal value-warm-up iteraties.
            num_rollout_steps: stappen per rollout per richting.
            noise_std: standaardafwijking van de ruis op expert-acties.
            bc_lr: learning rate voor BC.
        """
        rng = configuration.random.rng
        env = Environment(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)

        # Initialiseer model-parameters als dat nog niet gedaan is
        if self.params is None:
            rng, init_key = jax.random.split(rng)
            dummy_env = env.reset(init_key)
            dummy_input = self.build_obs_angle(dummy_env, 0.0)
            rng, init_key2 = jax.random.split(rng)
            self.params = self.model.init(init_key2, dummy_input)

        arm_angles = jnp.arange(5) * (2 * jnp.pi / 5)

        # Expert CPG-state voor arm 0 (rechts), roteer naar alle 5 richtingen
        expert_cpg_right = cpg_generator.modulate_body(cpg.reset(), body_cpg)
        expert_cpg_states = [
            cpg_generator.modulate_symmetric_rotation(expert_cpg_right, k)
            for k in range(5)
        ]
        # Expert acties: [frequency, amplitude_goals] per richting  (5, action_dim)
        expert_actions = jnp.stack([
            jnp.concatenate([jnp.atleast_1d(s.frequency), s.amplitude_goals])
            for s in expert_cpg_states
        ])

        expert_rollout_fn = self._make_expert_rollout_fn(
            env, cpg_generator, cpg, configuration, num_rollout_steps
        )

        # --- Fase 1: Behavioral Cloning ---
        bc_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(bc_lr))
        bc_opt_state = bc_optimizer.init(self.params)

        print("=== Fase 1: Behavioral Cloning ===")
        for iteration in range(bc_iterations):
            rng, *keys = jax.random.split(rng, 7)

            # Observaties verzamelen voor alle 5 richtingen
            obs_per_dir = [
                expert_rollout_fn(keys[k], expert_cpg_states[k], arm_angles[k])
                for k in range(5)
            ]
            obs_batch = jnp.concatenate(obs_per_dir)  # (5*num_rollout_steps, obs_dim)

            # Expert-acties herhalen per stap, kleine ruis toevoegen
            rng, noise_key = jax.random.split(rng)
            expert_targets = jnp.repeat(expert_actions, num_rollout_steps, axis=0)
            expert_targets = expert_targets + jax.random.normal(noise_key, expert_targets.shape) * noise_std

            self.params, bc_opt_state, loss = bc_update_jit(
                self.params, bc_opt_state, self.model, obs_batch, expert_targets, bc_optimizer
            )

            if iteration % 10 == 0:
                print(f"  BC iter {iteration:3d}, loss: {loss:.6f}")

        # --- Fase 2: Value warm-up ---
        print("=== Fase 2: Value warm-up ===")
        rollout_fn = self._make_rollout_fn(env, cpg_generator, self.model, configuration, num_rollout_steps)
        warmup_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
        warmup_opt_state = warmup_optimizer.init(self.params)

        for iteration in range(warmup_iterations):
            rng, subkey = jax.random.split(rng)
            keys = jax.random.split(subkey, 5)

            traj = jax.vmap(rollout_fn, in_axes=(0, None, 0))(keys, self.params, arm_angles)
            all_obs, all_act, all_logp, all_val, all_rew = traj

            _, last_vals = jax.vmap(lambda o: self.model.apply(self.params, o[-1]))(all_obs)
            val_bufs = jax.vmap(lambda v, lv: jnp.append(v, lv))(all_val, last_vals)

            advantages_list = [
                compute_gae(r, v, jnp.zeros(len(r)))
                for r, v in zip(all_rew, val_bufs)
            ]
            returns_list = [adv + v[:-1] for adv, v in zip(advantages_list, val_bufs)]
            returns = jnp.concatenate(returns_list)

            obs_buf = all_obs.reshape(-1, all_obs.shape[-1])
            act_buf = all_act.reshape(-1, all_act.shape[-1])
            logp_buf = all_logp.reshape(-1)
            dummy_adv = jnp.zeros(len(returns))

            batch = (obs_buf, act_buf, logp_buf, returns, dummy_adv)
            self.params, warmup_opt_state, loss = value_warmup_jit(
                self.params, warmup_opt_state, self.model, batch, warmup_optimizer
            )

            if iteration % 5 == 0:
                print(f"  Value warm-up iter {iteration:2d}, loss: {loss:.6f}")

        print("=== Pretraining klaar ===")
        return self.params

    def _make_rollout_fn(self, env, cpg_generator, model, configuration, num_steps):
        def rollout_fn(rng, params, angle):

            def scan_step(carry, _):
                cpg_state, env_state, prev_arm, rng = carry
                rng, subkey = jax.random.split(rng)

                x = self.build_obs_angle(env_state, angle)
                dist, value = model.apply(params, x)

                action = dist.sample(seed=subkey)
                log_prob = dist.log_prob(action)

                prev_pos = env_state.observations["disk_position"]
                prev_rot = env_state.observations["disk_rotation"][2]

                leading_arm_index = self.angle_to_arm_relative(
                    angle,
                    env_state.observations["disk_rotation"][2]
                )


                full_body = self.network_output_to_body(
                    action, cpg_state, leading_arm_index
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
                reward = self.angle_reward(prev_pos, curr_pos, prev_rot, curr_rot, angle) # - arm_switch_penalty

                return (cpg_state, env_state, leading_arm_index, rng), (
                    x, action, log_prob, value, reward
                )

            init = (
                cpg_generator.generate(configuration).reset(),
                env.reset(rng),
                self.angle_to_arm_relative(angle, 0.0),
                rng,
            )

            (_, _, _, _), traj = jax.lax.scan(scan_step, init, None, length=num_steps)
            return traj

        return jax.jit(rollout_fn)

    def save_controller(self, logger):
        path = Path(logger.base_folder) / "controller"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def read_controller(self, path: str):
        with open(path, "rb") as f:
            self.params = pickle.load(f)
            print(self.params["params"]["Dense_2"]["kernel"].shape[1])
            self.model = ActorCritic(
                action_dim=self.params["params"]["Dense_2"]["kernel"].shape[1]
            )

    def genome_size(self, configuration):
        flat_params, _ = jax.flatten_util.ravel_pytree(self.params)
        return flat_params.shape[0]
    

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