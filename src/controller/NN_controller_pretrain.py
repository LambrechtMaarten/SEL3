from pathlib import Path
from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from configs.config import Configuration
from src.controller.BaseNNController import BaseNNController
from src.cpg.cpg_generators.cpg_generators import CPGGenerator
from src.cpg.cpg_generators.fully_connected_symmetric_cpg_generator import (
    FullyConnectedSymmetricCPGGenerator,
)
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr

# === Behavioral Cloning ===


def bc_loss(params: Any, model: nn.Module, obs_batch: jarr, expert_actions: jarr) -> jarr:
    """MSE tussen policy-gemiddelde en expert-acties."""
    dist, _ = model.apply(params, obs_batch)
    return jnp.mean((dist.mean() - expert_actions) ** 2)


def bc_update_step(
    params: Any,
    opt_state: optax.OptState,
    model: nn.Module,
    obs_batch: jarr,
    expert_actions: jarr,
    optimizer: optax.GradientTransformation,
) -> Tuple[Any, optax.OptState, jarr]:
    loss, grads = jax.value_and_grad(bc_loss)(params, model, obs_batch, expert_actions)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


bc_update_jit = jax.jit(bc_update_step, static_argnames=["model", "optimizer"])


# === PPO met KL-regularisatie ===
def ppo_loss_with_kl(
    params: Any,
    model: nn.Module,
    batch: Tuple[jarr, jarr, jarr, jarr, jarr],
    pretrained_params: Any,
    kl_coef: float,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
):
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


def update_step_kl(
    params: Any,
    opt_state: optax.OptState,
    model: nn.Module,
    batch: Tuple[jarr, jarr, jarr, jarr, jarr],
    optimizer: optax.GradientTransformation,
    pretrained_params: Any,
    kl_coef: float,
) -> Tuple[Any, optax.OptState, jarr]:
    loss, grads = jax.value_and_grad(ppo_loss_with_kl)(
        params, model, batch, pretrained_params, kl_coef
    )
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


update_step_kl_jit = jax.jit(update_step_kl, static_argnames=["model", "optimizer"])

# === Value warm-up (enkel value head trainen) ===


def value_warmup_loss(
    params: Any,
    model: nn.Module,
    batch: Tuple[jarr, jarr, jarr, jarr, jarr],
) -> jarr:
    obs, _, _, returns, _ = batch
    _, values = model.apply(params, obs)
    return jnp.mean((returns - values) ** 2)


def value_warmup_step(
    params: Any,
    opt_state: optax.OptState,
    model: nn.Module,
    batch: Tuple[jarr, jarr, jarr, jarr, jarr],
    optimizer: optax.GradientTransformation,
) -> Tuple[Any, optax.OptState, jarr]:
    loss, grads = jax.value_and_grad(value_warmup_loss)(params, model, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


value_warmup_jit = jax.jit(value_warmup_step, static_argnames=["model", "optimizer"])


class NNControllerPretrain(BaseNNController):
    def __init__(self):
        super().__init__()  # 5 armen × 3 segmenten × 2 assen
        self.pretrained_params = None
        # KL-annealing: start bij 0.1, halveert elke ~35 iteraties
        self.kl_coef_init = jnp.array(0.1)
        self.kl_decay = jnp.array(50.0)
        self.target_angles = []

    def train_controller(self, configuration: Configuration, num_steps=1000, archive=None):
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
        self.logger = configuration.logger
        self.logger.init_logger()
        if archive is not None:
            self.pretrain_bc_from_archive(configuration, archive)
            self.pretrain_value_warmup(configuration)
            self.save_controller(self.logger, "pretrained_controller")

        self.pretrained_params = self.params

        # Leave rest of training to superclass
        super().train_controller(configuration, num_steps)

    def update_and_log(
        self,
        params: Any,
        opt_state: optax.OptState,
        batch: tuple[jarr, jarr, jarr, jarr, jarr],
        iteration: int,
        log_data: dict[str, Any],
    ) -> tuple[Any, optax.OptState]:
        kl_coef = self.kl_coef_init * jnp.exp(
            -jnp.array(iteration, dtype=jnp.float32) / self.kl_decay
        )
        for _ in range(self.epochs):
            params, opt_state, loss = update_step_kl_jit(
                params,
                opt_state,
                self.model,
                batch,
                self.optimizer,
                self.pretrained_params,
                kl_coef,
            )

        log_data["loss"] = loss
        log_data["kl_coef"] = kl_coef
        self.logger.log(log_data)
        return params, opt_state

    def _make_expert_rollout_fn(
        self,
        env: Environment,
        cpg_generator: CPGGenerator,
        cpg: CPGState,
        configuration: Configuration,
        num_steps: int,
    ) -> Callable:
        """Rollout waarbij de expert-CPG de robot bestuurt.
        Verzamelt observaties én de bijhorende joint actions voor BC."""

        def expert_rollout(rng, expert_cpg_state, angle, speed):
            def scan_step(carry, _):
                cpg_state, env_state = carry

                relative_angle = angle - env_state.observations["disk_rotation"][2]
                local_angle, k = self.to_local_angle_and_sector(relative_angle)

                obs = self.build_obs_angle(env_state.observations, local_angle, k, speed)

                cpg_state = cpg.step(cpg_state)
                joint_actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)

                shift = jnp.where(angle == jnp.pi / 5, -12, 0)
                joint_actions = jnp.roll(joint_actions, shift)

                env_state = env.step(joint_actions, env_state)
                return (cpg_state, env_state), (obs, joint_actions)

            init = (expert_cpg_state, env.reset(rng))
            (_, _), (obs_seq, action_seq) = jax.lax.scan(scan_step, init, None, length=num_steps)
            return obs_seq, action_seq  # (num_steps, obs_dim), (num_steps, 30)

        return jax.jit(expert_rollout)

    def pretrain_bc_from_archive(
        self,
        configuration: Configuration,
        archive_path: str,
        bc_iterations: int = 1000,
        num_rollout_steps: int = 500,
        noise_std: float = 0.02,
        bc_lr: float = 1e-3,
        min_displacement: float = 0.2,
        top_k: int = 15,
    ):
        """BC-pretraining met meerdere expert-gaits uit een map-elites archief.

        Vergelijkbaar met pretrain_bc, maar gebruikt top_k gaits uit het
        archief i.p.v. één handmatige expert-gait. Elke gait wordt naar
        5 richtingen geroteerd, wat tot top_k × 5 expert-rollouts leidt.

        Args:
            configuration: trainings-configuratie.
            archive_path: pad naar map met selections.npy en x_positions.npy.
            bc_iterations: aantal BC-updateslagen.
            num_rollout_steps: stappen per rollout per richting.
            noise_std: ruis op expert-acties voor generalisatie.
            bc_lr: learning rate voor BC.
            min_displacement: minimale |x|-verplaatsing om gait te gebruiken.
            top_k: aantal gaits (met grootste |x|) om te gebruiken.
        """

        rng = configuration.random.rng
        env = Environment(configuration)
        # Het archief is aangemaakt met de FullyConnectedSymmetricCPGGenerator (30 oscillatoren,
        # genome_size = 1 + 30 + 30 + 30×30 = 961).  We gebruiken die generator hier
        # expliciet, ongeacht de CPG-configuratie in `configuration`.
        cpg_generator = FullyConnectedSymmetricCPGGenerator()
        cpg = cpg_generator.generate(configuration)

        # Initialiseer model-parameters
        if self.params is None:
            rng, init_key = jax.random.split(rng)
            dummy_env = env.reset(init_key)
            dummy_input = self.build_obs_angle(dummy_env.observations, 0.0)
            rng, init_key2 = jax.random.split(rng)
            self.params = self.model.init(init_key2, dummy_input)

        # Laad archief
        archive = Path(archive_path)
        selections = np.load(archive / "selections.npy")  # (N, genome_size)
        x_positions = np.load(archive / "x_positions.npy")  # (N,)

        # Filter op gaits die duidelijk bewegen
        mask = jnp.abs(x_positions) > min_displacement
        filtered_selections = selections[mask]
        filtered_x = x_positions[mask]

        # Uniforme sampling over het snelheidsbereik: top_k even verdeelde bins,
        # één representant per bin (de snelste in dat interval).
        max_speed = float(np.max(np.abs(filtered_x)))
        speed_bins = np.linspace(min_displacement, max_speed, top_k + 1)
        selected_indices = []
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            in_bin = np.where((np.abs(filtered_x) >= lo) & (np.abs(filtered_x) < hi))[0]
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

        print(
            f"Archief: {len(selections)} gaits, {k} geselecteerd "
            f"(uniform over [{min_displacement:.2f}, {max_speed:.2f}], {top_k} bins)."
        )

        expert_rollout_fn = self._make_expert_rollout_fn(
            env, cpg_generator, cpg, configuration, num_rollout_steps
        )

        # --- Fase 1: Behavioral Cloning ---
        # Genereer expert-data EENMALIG — rollouts zijn deterministisch voor
        # vaste CPG-params, dus elke iteratie opnieuw genereren is pure verspilling.
        print("Expert-rollouts genereren (eenmalig)...")
        rng, *keys = jax.random.split(rng, k + 1)
        all_obs = []
        all_actions = []
        n_sim_steps = 800

        for gait_idx, (gait_params, x_pos) in enumerate(zip(top_selections, top_x)):
            norm_speed = float(abs(x_pos) / max_speed)
            self.norm_speeds.append(norm_speed)
            self.speeds.append(abs(x_pos) / n_sim_steps)

            target_angle = 0.0 if x_pos > 0 else jnp.pi / 5
            self.target_angles.append(target_angle)

            expert_cpg_state = cpg_generator.modulate_body(cpg.reset(), jnp.array(gait_params))

            obs, actions = expert_rollout_fn(
                keys[gait_idx], expert_cpg_state, target_angle, norm_speed
            )

            all_obs.append(obs)
            all_actions.append(actions)
            direction = "pos" if x_pos > 0 else "neg"
            print(f"  Gait {gait_idx + 1}/{k} ({direction}) snelheid={norm_speed:.2f}  klaar")

        obs_dataset = jnp.concatenate(all_obs)  # (k*5*T, obs_dim)
        target_dataset = jnp.concatenate(all_actions)  # (k*5*T, 30)

        # Voeg eenmalig ruis toe voor generalisatie
        rng, noise_key = jax.random.split(rng)
        target_dataset = (
            target_dataset + jax.random.normal(noise_key, target_dataset.shape) * noise_std
        )

        n_samples = obs_dataset.shape[0]
        batch_size = 512
        print(
            f"Dataset: {n_samples} samples — {bc_iterations} epochs, mini-batches van {batch_size}"
        )

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
                idx = perm[start : start + batch_size]
                self.params, bc_opt_state, loss = bc_update_jit(
                    self.params,
                    bc_opt_state,
                    self.model,
                    obs_dataset[idx],
                    target_dataset[idx],
                    bc_optimizer,
                )
                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if self.logger is not None:
                self.logger.log(
                    {
                        "pretrain/bc_loss": avg_loss,
                        "pretrain/bc_iteration": iteration,
                    }
                )

    def pretrain_value_warmup(
        self,
        configuration: Configuration,
        num_rollout_steps: int = 500,
        warmup_iterations: int = 20,
    ):
        rng = configuration.random.rng
        env = Environment(configuration)

        print("=== Fase 2: Value warm-up ===")
        rollout_fn = self._make_rollout_fn(env, self.model, configuration, num_rollout_steps)
        warmup_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
        warmup_opt_state = warmup_optimizer.init(self.params)

        for iteration in range(warmup_iterations):
            rng, subkey = jax.random.split(rng)

            keys = jax.random.split(subkey, len(self.speeds))

            traj = jax.vmap(rollout_fn, in_axes=(0, None, 0, 0, 0))(
                keys,
                self.params,
                jnp.array(self.target_angles),
                jnp.array(self.speeds),
                jnp.array(self.norm_speeds),
            )

            all_obs, all_act, all_logp, all_val, all_rew, _, _ = traj

            _, last_vals = jax.vmap(lambda o: self.model.apply(self.params, o[-1]))(all_obs)
            val_bufs = jax.vmap(lambda v, lv: jnp.append(v, lv))(all_val, last_vals)

            advantages_list = [
                self.compute_gae(r, v, jnp.zeros(len(r))) for r, v in zip(all_rew, val_bufs)
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
                self.logger.log(
                    {
                        "pretrain/warmup_loss": float(loss),
                        "pretrain/warmup_iteration": iteration,
                    }
                )

        print("=== Pretraining vanuit archief klaar ===")
        return self.params
