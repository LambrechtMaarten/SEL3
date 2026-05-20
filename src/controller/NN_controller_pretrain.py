"""Neural-network controller with Behavioral Cloning pre-training from a map-elites archive."""

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
    """Compute the Behavioral Cloning loss (MSE between policy mean and expert actions).

    Args:
        params: Current Flax model parameters.
        model: The ActorCritic Flax module.
        obs_batch: Batch of observations, shape ``(batch_size, obs_dim)``.
        expert_actions: Corresponding expert joint actions, shape
            ``(batch_size, action_dim)``.

    Returns:
        Scalar mean-squared-error loss between the policy mean and the expert
        actions.
    """
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
    """Perform a single Behavioral Cloning gradient update step.

    Args:
        params: Current Flax model parameters.
        opt_state: Current Optax optimizer state.
        model: The ActorCritic Flax module.
        obs_batch: Batch of observations, shape ``(batch_size, obs_dim)``.
        expert_actions: Corresponding expert joint actions, shape
            ``(batch_size, action_dim)``.
        optimizer: Optax gradient transformation to apply.

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value).
    """
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
    """Compute the PPO loss with a KL-divergence regularisation term.

    An additional KL-divergence penalty keeps the current policy close to the
    pre-trained initialisation.  The coefficient ``kl_coef`` is annealed
    (decreased) over the course of training.

    Args:
        params: Current Flax model parameters.
        model: The ActorCritic Flax module.
        batch: Tuple of (observations, actions, old_log_probs, returns,
            advantages).
        pretrained_params: Flax parameters of the BC-pre-trained policy used
            as the KL reference distribution.
        kl_coef: Weight of the KL-divergence penalty term.
        clip_eps: PPO clipping epsilon for the policy ratio.
        vf_coef: Coefficient for the value function loss term.
        ent_coef: Coefficient for the entropy bonus term.

    Returns:
        Scalar combined loss value.
    """
    obs, actions, old_log_probs, returns, advantages = batch

    dist, values = model.apply(params, obs)
    pretrained_dist, _ = model.apply(pretrained_params, obs)

    log_probs = dist.log_prob(actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    value_loss = jnp.mean((returns - values) ** 2)
    entropy = jnp.mean(dist.entropy())

    # KL(current policy || pre-trained policy)
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
    """Perform a single gradient update step using the KL-regularised PPO loss.

    Args:
        params: Current Flax model parameters.
        opt_state: Current Optax optimizer state.
        model: The ActorCritic Flax module.
        batch: Tuple of (observations, actions, old_log_probs, returns,
            advantages).
        optimizer: Optax gradient transformation to apply.
        pretrained_params: Flax parameters of the BC-pre-trained policy.
        kl_coef: Current annealed KL-divergence penalty coefficient.

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value).
    """
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
    """Compute the value head warm-up loss (MSE between predicted and Monte-Carlo returns).

    Only the value head is effectively trained here because the policy head
    gradients are not used during the warm-up phase.

    Args:
        params: Current Flax model parameters.
        model: The ActorCritic Flax module.
        batch: Tuple of (observations, actions, old_log_probs, returns,
            advantages).  Only observations and returns are used.

    Returns:
        Scalar mean-squared-error loss between predicted values and returns.
    """
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
    """Perform a single gradient update step for the value head warm-up phase.

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
    loss, grads = jax.value_and_grad(value_warmup_loss)(params, model, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


value_warmup_jit = jax.jit(value_warmup_step, static_argnames=["model", "optimizer"])


class NNControllerPretrain(BaseNNController):
    """Neural-network controller with Behavioral Cloning pre-training.

    Extends :class:`BaseNNController` with a two-phase training procedure:

    1. **Behavioral Cloning (BC)** — the policy is initialised by imitating
       expert CPG gaits drawn from a map-elites archive.
    2. **Value warm-up** — the value head is calibrated on rollouts collected
       with the BC-initialised policy.
    3. **KL-regularised PPO** — standard PPO training with an additional KL
       penalty that keeps the policy close to the BC initialisation.  The
       penalty coefficient is annealed exponentially during training.

    Attributes:
        pretrained_params: Snapshot of the Flax parameters after BC
            pre-training, used as the KL reference distribution.
        kl_coef_init: Initial KL penalty coefficient.
        kl_decay: Exponential decay constant (in iterations) for KL annealing.
        target_angles: List of target heading angles collected from the
            archive gaits during BC pre-training.
    """

    def __init__(self):
        """Initialise NNControllerPretrain with KL annealing hyperparameters."""
        super().__init__()  # 5 arms × 3 segments × 2 axes
        self.pretrained_params = None
        # KL annealing: starts at 0.1, halves approximately every 35 iterations
        self.kl_coef_init = jnp.array(0.1)
        self.kl_decay = jnp.array(50.0)
        self.target_angles = []

    def train_controller(self, configuration: Configuration, num_steps=1000, archive=None):
        """Train the controller using BC pre-training followed by KL-regularised PPO.

        If an archive path is provided, the training proceeds in three stages:
        (1) Behavioral Cloning from map-elites expert gaits, (2) value head
        warm-up, and (3) KL-regularised PPO via the parent class.  When no
        archive is given, training proceeds directly with PPO.

        Args:
            configuration: Global simulation and training configuration.
            num_steps: Number of simulation steps per rollout trajectory.
            archive: Optional filesystem path to a map-elites archive
                directory containing ``selections.npy`` and
                ``x_positions.npy``.  When provided, BC pre-training is run
                before PPO.
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
        """Run PPO update epochs with KL annealing, log metrics, and return updated state.

        Overrides the base-class method to use the KL-regularised loss and
        log the current KL coefficient alongside the standard metrics.

        Args:
            params: Current Flax model parameters.
            opt_state: Current Optax optimizer state.
            batch: Tuple of (observations, actions, old_log_probs, returns,
                advantages).
            iteration: Current PPO iteration index, used to compute the
                annealed KL coefficient.
            log_data: Dictionary of scalar metrics to be logged.

        Returns:
            Tuple of (updated_params, updated_opt_state).
        """
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
        """Build a JIT-compiled expert rollout function driven by a CPG.

        The expert policy is a CPG-based gait rather than the neural network.
        The function collects observations together with the corresponding
        joint actions produced by the CPG, which are used as supervision
        signal for Behavioral Cloning.

        Args:
            env: Wrapped MuJoCo environment.
            cpg_generator: CPG generator used to map CPG outputs to joint
                actions.
            cpg: CPG instance whose ``step`` method advances the oscillators.
            configuration: Global simulation and training configuration.
            num_steps: Number of simulation steps to collect per episode.

        Returns:
            JIT-compiled callable with signature
            ``(rng, expert_cpg_state, angle, speed) -> (obs_seq, action_seq)``.
        """

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
        """Pre-train the policy via Behavioral Cloning using expert gaits from a map-elites archive.

        Selects up to ``top_k`` gaits uniformly distributed across the speed
        range found in the archive, collects expert rollouts for each, and
        trains the policy with mini-batch gradient descent to minimise the MSE
        between the policy mean and the expert joint actions.

        The archive is expected to have been created with
        :class:`FullyConnectedSymmetricCPGGenerator` (genome size = 961).

        Args:
            configuration: Global simulation and training configuration.
            archive_path: Path to the directory containing ``selections.npy``
                and ``x_positions.npy`` produced by :func:`train_archive`.
            bc_iterations: Number of full passes (epochs) over the expert
                dataset.
            num_rollout_steps: Number of simulation steps per expert rollout.
            noise_std: Standard deviation of Gaussian noise added to expert
                actions to improve generalisation.
            bc_lr: Learning rate for the BC Adam optimizer.
            min_displacement: Minimum absolute x-displacement for a gait to
                be included in the training set.
            top_k: Maximum number of gaits to select (one per speed bin).
        """

        rng = configuration.random.rng
        env = Environment(configuration)
        # The archive was created with FullyConnectedSymmetricCPGGenerator (30 oscillators,
        # genome_size = 1 + 30 + 30 + 30×30 = 961).  We use that generator explicitly here,
        # regardless of the CPG configuration in `configuration`.
        cpg_generator = FullyConnectedSymmetricCPGGenerator()
        cpg = cpg_generator.generate(configuration)

        # Initialise model parameters
        if self.params is None:
            rng, init_key = jax.random.split(rng)
            dummy_env = env.reset(init_key)
            dummy_input = self.build_obs_angle(dummy_env.observations, 0.0)
            rng, init_key2 = jax.random.split(rng)
            self.params = self.model.init(init_key2, dummy_input)

        # Load archive
        archive = Path(archive_path)
        selections = np.load(archive / "selections.npy")  # (N, genome_size)
        x_positions = np.load(archive / "x_positions.npy")  # (N,)

        # Filter to gaits with meaningful displacement
        mask = jnp.abs(x_positions) > min_displacement
        filtered_selections = selections[mask]
        filtered_x = x_positions[mask]

        # Uniform sampling over the speed range: top_k evenly spaced bins,
        # one representative per bin (the fastest gait in that interval).
        max_speed = float(np.max(np.abs(filtered_x)))
        speed_bins = np.linspace(min_displacement, max_speed, top_k + 1)
        selected_indices = []
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            in_bin = np.where((np.abs(filtered_x) >= lo) & (np.abs(filtered_x) < hi))[0]
            if len(in_bin) > 0:
                best = in_bin[np.argmax(np.abs(filtered_x[in_bin]))]
                selected_indices.append(int(best))

        # Always include the fastest gait if it is not already selected
        fastest_idx = int(np.argmax(np.abs(filtered_x)))
        if fastest_idx not in selected_indices:
            selected_indices.append(fastest_idx)

        top_selections = filtered_selections[selected_indices]
        top_x = filtered_x[selected_indices]
        k = len(selected_indices)

        print(
            f"Archive: {len(selections)} gaits, {k} selected "
            f"(uniform over [{min_displacement:.2f}, {max_speed:.2f}], {top_k} bins)."
        )

        expert_rollout_fn = self._make_expert_rollout_fn(
            env, cpg_generator, cpg, configuration, num_rollout_steps
        )

        # --- Phase 1: Behavioral Cloning ---
        # Generate expert data ONCE — rollouts are deterministic for fixed CPG
        # parameters, so regenerating every iteration would be wasteful.
        print("Generating expert rollouts (one-time)...")
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
            print(f"  Gait {gait_idx + 1}/{k} ({direction}) speed={norm_speed:.2f}  done")

        obs_dataset = jnp.concatenate(all_obs)  # (k*5*T, obs_dim)
        target_dataset = jnp.concatenate(all_actions)  # (k*5*T, 30)

        # Add noise once for generalisation
        rng, noise_key = jax.random.split(rng)
        target_dataset = (
            target_dataset + jax.random.normal(noise_key, target_dataset.shape) * noise_std
        )

        n_samples = obs_dataset.shape[0]
        batch_size = 512
        print(
            f"Dataset: {n_samples} samples — {bc_iterations} epochs, mini-batches of {batch_size}"
        )

        bc_optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(bc_lr))
        bc_opt_state = bc_optimizer.init(self.params)

        print("=== Phase 1: Behavioral Cloning (archive) ===")
        for iteration in range(bc_iterations):
            # Shuffle and iterate in mini-batches
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
        """Warm up the value head using rollouts collected with the BC-initialised policy.

        Runs several rollout iterations with the current (BC-pre-trained)
        policy, computes Monte-Carlo returns, and trains only the value head
        to match those returns.  This gives the value function a reasonable
        initialisation before full PPO begins.

        Args:
            configuration: Global simulation and training configuration.
            num_rollout_steps: Number of simulation steps per rollout.
            warmup_iterations: Number of value warm-up iterations.

        Returns:
            Updated Flax model parameters after value warm-up.
        """
        rng = configuration.random.rng
        env = Environment(configuration)

        print("=== Phase 2: Value warm-up ===")
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

        print("=== Pre-training from archive complete ===")
        return self.params
