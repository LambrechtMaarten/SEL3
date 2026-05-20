"""Genetic optimizer with log-scale random normal mutation for adaptive step sizes."""

from typing import Any, Tuple

import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class RandomNormalGeneticOptimizer(GeneticOptimizer):
    """Genetic optimizer using truncation selection and per-gene log-scale mutation.

    Selection keeps the top 50 % of individuals.  Reproduction perturbs each
    gene independently with Gaussian noise whose standard deviation is drawn
    log-uniformly from [0.001, ~2.5], allowing both small refinements and
    large jumps in parameter space.
    """

    def select(self, population: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Select the top 50 % of individuals by fitness score.

        Supports both scalar fitness arrays and tuple evaluations (uses the
        first element of the tuple as the score).

        Args:
            population: Genome array of shape ``(population_size, genome_size)``.
            evaluations: Fitness scores or tuple of score arrays.
            rng: Unused.
            state: Unused.

        Returns:
            Tuple of (top_half_genomes, None).
        """
        scores = evaluations[0] if isinstance(evaluations, tuple) else evaluations
        return population[jnp.argsort(scores)[-jnp.size(population, 0) // 2 :]], None

    def reproduce(self, genomes: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Create offspring by applying per-gene log-scale Gaussian mutation.

        Each gene is perturbed by N(0, sigma) where sigma is sampled
        log-uniformly per gene from [0.001, ~2.5].

        Args:
            genomes: Selected survivor genomes of shape
                ``(n_survivors, genome_size)``.
            evaluations: Unused.
            rng: JAX random key.
            state: Unused.

        Returns:
            Tuple of (new_population, None) concatenating survivors and
            offspring.
        """
        rng1, rng2 = jax.random.split(rng)
        multiplier = 0.01 * (
            10 ** jax.random.uniform(rng1, minval=-1, maxval=2.4, shape=genomes.shape)
        )
        return jnp.concatenate(
            [genomes, genomes + jax.random.normal(rng2, genomes.shape) * multiplier]
        ), None
