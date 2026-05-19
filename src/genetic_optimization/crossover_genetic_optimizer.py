"""Genetic optimizer that uses arm-aware crossover and log-scale Gaussian mutation."""

from typing import Any, Tuple

import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class CrossoverGeneticOptimizer(GeneticOptimizer):
    """Genetic optimizer with arm-structured crossover and adaptive mutation.

    Selection keeps the top 50 % of individuals by fitness.  Reproduction
    pairs survivors randomly and performs arm-level crossover: for each of
    the 5 arms a Bernoulli mask selects whether the child inherits amplitude,
    offset, and phase-bias parameters from parent 1 or parent 2.  Mutation
    uses log-uniformly sampled step sizes (0.01 to ~2.5) so that both fine
    and coarse perturbations are explored.
    """

    def select(self, population: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Select the top 50 % of individuals by fitness score.

        Args:
            population: Genome array of shape ``(population_size, genome_size)``.
            evaluations: Fitness scores of shape ``(population_size,)``.
            rng: Unused.
            state: Unused.

        Returns:
            Tuple of (top_half_genomes, None).
        """
        n_select = jnp.size(population, 0) // 2
        selected_idx = jnp.argsort(evaluations)[-n_select:]
        return population[selected_idx], None

    def reproduce(self, genomes: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Generate offspring via arm-level crossover followed by adaptive mutation.

        Args:
            genomes: Selected survivor genomes of shape
                ``(n_survivors, genome_size)``.
            evaluations: Unused.
            rng: JAX random key.
            state: Unused.

        Returns:
            Tuple of (new_population, None) where new_population concatenates
            the original survivors with their crossed-over, mutated offspring.
        """
        n_genomes = genomes.shape[0]

        rng, rng_perm, rng_armmask, rng_mutation = jax.random.split(rng, 4)

        # parent pairing
        perm = jax.random.permutation(rng_perm, n_genomes)
        parents1 = genomes
        parents2 = genomes[perm]

        # ---- genome structure ----
        num_arms = 5
        oscillators_per_arm = 2
        n_osc = num_arms * oscillators_per_arm

        freq_size = 1
        amp_size = n_osc
        off_size = n_osc
        phase_size = n_osc * n_osc

        # split genome into its structural components
        def split_genome(g):
            freq = g[:, :freq_size]
            amp = g[:, freq_size : freq_size + amp_size]
            off = g[:, freq_size + amp_size : freq_size + amp_size + off_size]
            phase = g[:, -phase_size:].reshape((-1, n_osc, n_osc))
            return freq, amp, off, phase

        f1, a1, o1, p1 = split_genome(parents1)
        f2, a2, o2, p2 = split_genome(parents2)

        # ---- arm mask ----
        arm_mask = jax.random.bernoulli(rng_armmask, 0.5, (n_genomes, num_arms))

        # expand mask for oscillator parameters
        osc_mask = jnp.repeat(arm_mask, oscillators_per_arm, axis=1)

        # crossover amplitude + offset
        child_amp = jnp.where(osc_mask, a1, a2)
        child_off = jnp.where(osc_mask, o1, o2)

        # ---- phase bias crossover (block per arm pair) ----
        phase_mask = jnp.repeat(osc_mask[:, :, None], n_osc, axis=2)
        child_phase = jnp.where(phase_mask, p1, p2)

        # ---- frequency crossover ----
        freq_mask = jax.random.bernoulli(rng_armmask, 0.5, (n_genomes, 1))
        child_freq = jnp.where(freq_mask, f1, f2)

        # ---- flatten genome again ----
        children = jnp.concatenate(
            [child_freq, child_amp, child_off, child_phase.reshape(n_genomes, -1)],
            axis=1,
        )

        # ---- mutation ----
        multiplier = 0.01 * (
            10 ** jax.random.uniform(rng_mutation, minval=0, maxval=2.4, shape=children.shape)
        )

        mutation = jax.random.normal(rng_mutation, children.shape) * multiplier
        children = children + mutation

        new_population = jnp.concatenate([genomes, children])

        return new_population, None
