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
from src.cpg.cpg_generators.fully_connected_cpg_generator import FullyConnectedCPGGenerator
from src.environment.environment import Environment
from src.controller.BaseNNController import BaseNNController

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

class NNControllerPretrain(BaseNNController):
    def __init__(self):
        super().__init__(30)  # 5 armen × 3 segmenten × 2 assen
        self.pretrained_params = None
        # KL-annealing: start bij 0.1, halveert elke ~35 iteraties
        self.kl_coef_init = jnp.array(0.1)
        self.kl_decay = jnp.array(50.0)

    def train_controller(self, configuration, num_steps=1000, pretrained_body_cpg=None):
        """Train de controller via PPO.

        Args:
            configuration: trainings-configuratie.
            num_steps: stappen per rollout.
            epochs: PPO-epochs per iteratie.
            pretrained_body_cpg: optionele platte CPG-array van een expert-gait.
                Als opgegeven, wordt eerst Behavioral Cloning + value warm-up
                uitgevoerd, gevolgd door PPO met KL-regularisatie.
        """
        # === Pretraining via Behavioral Cloning ===
        if pretrained_body_cpg is not None:
            self.pretrain_bc(configuration, pretrained_body_cpg)
            self.pretrained_params = self.params
            params = self.params

        # Leave rest of training to superclass
        super().train_controller(configuration, num_steps)
    
    def update_and_log(self, params, opt_state, batch, iteration, log_data):
        kl_coef = self.kl_coef_init * jnp.exp(-jnp.array(iteration, dtype=jnp.float32) / self.kl_decay)
        for _ in range(self.epochs):
            params, opt_state, loss = update_step_kl_jit(
                params, opt_state, self.model, batch, self.optimizer, self.pretrained_params, kl_coef
            )
    
        log_data["loss"] = loss
        log_data["kl_coef"] = kl_coef
        self.logger.log(log_data)
        print(f"Iter {iteration}, loss {loss}")
        return params, opt_state

    def _make_expert_rollout_fn(self, env, cpg_generator, cpg, configuration, num_steps):
        """Rollout waarbij de expert-CPG de robot bestuurt.
        Verzamelt observaties én de bijhorende joint actions voor BC."""
        def expert_rollout(rng, expert_cpg_state, angle, speed):
            def scan_step(carry, _):
                cpg_state, env_state = carry
                obs = self.build_obs_angle(env_state, angle, speed)
                cpg_state = cpg.step(cpg_state)
                joint_actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
                env_state = env.step(joint_actions, env_state)
                return (cpg_state, env_state), (obs, joint_actions)

            init = (expert_cpg_state, env.reset(rng))
            (_, _), (obs_seq, action_seq) = jax.lax.scan(scan_step, init, None, length=num_steps)
            return obs_seq, action_seq  # (num_steps, obs_dim), (num_steps, 30)

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

        expert_rollout_fn = self._make_expert_rollout_fn(
            env, cpg_generator, cpg, configuration, num_rollout_steps
        )

        # --- Fase 1: Behavioral Cloning ---
        bc_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(bc_lr))
        bc_opt_state = bc_optimizer.init(self.params)

        print("=== Fase 1: Behavioral Cloning ===")
        for iteration in range(bc_iterations):
            rng, *keys = jax.random.split(rng, 7)

            # Observaties én joint actions verzamelen voor alle 5 richtingen (speed=1.0)
            rollouts = [
                expert_rollout_fn(keys[k], expert_cpg_states[k], arm_angles[k], 1.0)
                for k in range(5)
            ]
            obs_batch = jnp.concatenate([r[0] for r in rollouts])       # (5*num_rollout_steps, obs_dim)
            expert_targets = jnp.concatenate([r[1] for r in rollouts])  # (5*num_rollout_steps, 30)

            # Kleine ruis toevoegen voor generalisatie
            rng, noise_key = jax.random.split(rng)
            expert_targets = expert_targets + jax.random.normal(noise_key, expert_targets.shape) * noise_std

            self.params, bc_opt_state, loss = bc_update_jit(
                self.params, bc_opt_state, self.model, obs_batch, expert_targets, bc_optimizer
            )

            if iteration % 10 == 0:
                print(f"  BC iter {iteration:3d}, loss: {loss:.6f}")

        # --- Fase 2: Value warm-up ---
        print("=== Fase 2: Value warm-up ===")
        rollout_fn = self._make_rollout_fn(env, self.model, configuration, num_rollout_steps)
        warmup_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
        warmup_opt_state = warmup_optimizer.init(self.params)

        for iteration in range(warmup_iterations):
            rng, subkey = jax.random.split(rng)
            keys = jax.random.split(subkey, 5)
            speeds = jnp.ones(5)

            traj = jax.vmap(rollout_fn, in_axes=(0, None, 0, 0))(keys, self.params, arm_angles, speeds)
            all_obs, all_act, all_logp, all_val, all_rew = traj

            _, last_vals = jax.vmap(lambda o: self.model.apply(self.params, o[-1]))(all_obs)
            val_bufs = jax.vmap(lambda v, lv: jnp.append(v, lv))(all_val, last_vals)

            advantages_list = [
                self.compute_gae(r, v, jnp.zeros(len(r)))
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

    def pretrain_bc_from_archive(self, configuration, archive_path,
                                  bc_iterations=100, warmup_iterations=20,
                                  num_rollout_steps=500, noise_std=0.02,
                                  bc_lr=1e-3, min_displacement=0.2, top_k=10):
        """BC-pretraining met meerdere expert-gaits uit een map-elites archief.

        Vergelijkbaar met pretrain_bc, maar gebruikt top_k gaits uit het
        archief i.p.v. één handmatige expert-gait. Elke gait wordt naar
        5 richtingen geroteerd, wat tot top_k × 5 expert-rollouts leidt.

        Args:
            configuration: trainings-configuratie.
            archive_path: pad naar map met selections.npy en x_positions.npy.
            bc_iterations: aantal BC-updateslagen.
            warmup_iterations: aantal value warm-up iteraties.
            num_rollout_steps: stappen per rollout per richting.
            noise_std: ruis op expert-acties voor generalisatie.
            bc_lr: learning rate voor BC.
            min_displacement: minimale |x|-verplaatsing om gait te gebruiken.
            top_k: aantal gaits (met grootste |x|) om te gebruiken.
        """
        import numpy as np
        from pathlib import Path

        rng = configuration.random.rng
        env = Environment(configuration)
        # Het archief is aangemaakt met de FullyConnectedCPGGenerator (30 oscillatoren,
        # genome_size = 1 + 30 + 30 + 30×30 = 961).  We gebruiken die generator hier
        # expliciet, ongeacht de CPG-configuratie in `configuration`.
        cpg_generator = FullyConnectedCPGGenerator()
        cpg = cpg_generator.generate(configuration)

        # Initialiseer model-parameters
        if self.params is None:
            rng, init_key = jax.random.split(rng)
            dummy_env = env.reset(init_key)
            dummy_input = self.build_obs_angle(dummy_env, 0.0)
            rng, init_key2 = jax.random.split(rng)
            self.params = self.model.init(init_key2, dummy_input)

        # Laad archief
        archive = Path(archive_path)
        selections = np.load(archive / "selections.npy")    # (N, genome_size)
        x_positions = np.load(archive / "x_positions.npy") # (N,)

        # Filter op gaits die duidelijk bewegen
        mask = np.abs(x_positions) > min_displacement
        filtered_selections = selections[mask]
        filtered_x = x_positions[mask]

        # Uniforme sampling over het snelheidsbereik: top_k even verdeelde bins,
        # één representant per bin (de snelste in dat interval).
        max_speed = float(np.max(np.abs(filtered_x)))
        speed_bins = np.linspace(min_displacement, max_speed, top_k + 1)
        selected_indices = []
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            in_bin = np.where(
                (np.abs(filtered_x) >= lo) & (np.abs(filtered_x) < hi)
            )[0]
            if len(in_bin) > 0:
                best = in_bin[np.argmax(np.abs(filtered_x[in_bin]))]
                selected_indices.append(int(best))
        # Voeg altijd de snelste gait toe als die nog niet in de selectie zit
        fastest_idx = int(np.argmax(np.abs(filtered_x)))
        if fastest_idx not in selected_indices:
            selected_indices.append(fastest_idx)

        top_selections = filtered_selections[selected_indices]
        top_x = filtered_x[selected_indices]
        k = len(selected_indices)

        print(f"Archief: {len(selections)} gaits, {k} geselecteerd "
              f"(uniform over [{min_displacement:.2f}, {max_speed:.2f}], {top_k} bins).")

        arm_angles = jnp.arange(5) * (2 * jnp.pi / 5)
        expert_rollout_fn = self._make_expert_rollout_fn(
            env, cpg_generator, cpg, configuration, num_rollout_steps
        )

        # --- Fase 1: Behavioral Cloning ---
        # Genereer expert-data EENMALIG — rollouts zijn deterministisch voor
        # vaste CPG-params, dus elke iteratie opnieuw genereren is pure verspilling.
        print("Expert-rollouts genereren (eenmalig)...")
        rng, *keys = jax.random.split(rng, k * 5 + 1)
        key_idx = 0
        all_obs = []
        all_actions = []

        for gait_idx, (gait_params, x_pos) in enumerate(zip(top_selections, top_x)):
            norm_speed = float(abs(x_pos) / max_speed)
            base_rot = 0 if x_pos > 0 else len(arm_angles) // 2
            expert_cpg = cpg_generator.modulate_body(cpg.reset(), jnp.array(gait_params))
            expert_cpg_states = [
                cpg_generator.modulate_symmetric_rotation(expert_cpg, (i + base_rot) % len(arm_angles))
                for i in range(len(arm_angles))
            ]
            rollouts = [
                expert_rollout_fn(keys[key_idx + i], expert_cpg_states[i], arm_angles[i], norm_speed)
                for i in range(len(arm_angles))
            ]
            key_idx += len(arm_angles)
            all_obs.extend([r[0] for r in rollouts])
            all_actions.extend([r[1] for r in rollouts])
            print(f"  Gait {gait_idx + 1}/{k}  snelheid={norm_speed:.2f}  klaar")

        obs_dataset = jnp.concatenate(all_obs)       # (k*5*T, obs_dim)
        target_dataset = jnp.concatenate(all_actions) # (k*5*T, 30)

        # Voeg eenmalig ruis toe voor generalisatie
        rng, noise_key = jax.random.split(rng)
        target_dataset = target_dataset + jax.random.normal(
            noise_key, target_dataset.shape) * noise_std

        n_samples = obs_dataset.shape[0]
        batch_size = 2048
        print(f"Dataset: {n_samples} samples — {bc_iterations} epochs, "
              f"mini-batches van {batch_size}")

        bc_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(bc_lr))
        bc_opt_state = bc_optimizer.init(self.params)

        print("=== Fase 1: Behavioral Cloning (archief) ===")
        for iteration in range(bc_iterations):
            # Shuffle en doorloop in mini-batches
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                self.params, bc_opt_state, loss = bc_update_jit(
                    self.params, bc_opt_state, self.model,
                    obs_dataset[idx], target_dataset[idx], bc_optimizer
                )
                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if self.logger is not None:
                self.logger.log({
                    "pretrain/bc_loss": avg_loss,
                    "pretrain/bc_iteration": iteration,
                })
            if iteration % 10 == 0:
                print(f"  BC epoch {iteration:3d}, loss: {avg_loss:.6f}")

        # --- Fase 2: Value warm-up ---
        print("=== Fase 2: Value warm-up ===")
        rollout_fn = self._make_rollout_fn(env, self.model, configuration, num_rollout_steps)
        warmup_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
        warmup_opt_state = warmup_optimizer.init(self.params)

        for iteration in range(warmup_iterations):
            rng, subkey = jax.random.split(rng)
            keys = jax.random.split(subkey, 5)
            speeds = jnp.ones(5)  # volle snelheid voor warm-up

            traj = jax.vmap(rollout_fn, in_axes=(0, None, 0, 0))(keys, self.params, arm_angles, speeds)
            all_obs, all_act, all_logp, all_val, all_rew = traj

            _, last_vals = jax.vmap(lambda o: self.model.apply(self.params, o[-1]))(all_obs)
            val_bufs = jax.vmap(lambda v, lv: jnp.append(v, lv))(all_val, last_vals)

            advantages_list = [
                self.compute_gae(r, v, jnp.zeros(len(r)))
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

            if self.logger is not None:
                self.logger.log({
                    "pretrain/warmup_loss": float(loss),
                    "pretrain/warmup_iteration": iteration,
                })

            if iteration % 5 == 0:
                print(f"  Value warm-up iter {iteration:2d}, loss: {loss:.6f}")

        print("=== Pretraining vanuit archief klaar ===")
        return self.params

    def act(self, cpg_state, control_input, configuration, env_state):
        STOP_THRESHOLD = 0.05

        obs = env_state.observations
        robot_pos = obs["disk_position"][0:2]
        deltas = jnp.array(control_input) - robot_pos
        distance = jnp.linalg.norm(deltas)

        # Doel bereikt: geen beweging
        if distance < STOP_THRESHOLD:
            return jnp.zeros(30)

        angle = jnp.arctan2(deltas[1], deltas[0])
        rot = obs["disk_rotation"][2]

        # Snelheid: proportioneel aan afstand tot doel, verzadigt op 1.0
        # (robot vertraagt automatisch als het doel nadert)
        speed = jnp.clip(distance, 0.0, 1.0)

        x = self.build_obs_angle(env_state, angle - rot, speed)
        dist, _ = self.model.apply(self.params, x)

        # Directe joint actions — geen CPG tussenstap
        return dist.mode()