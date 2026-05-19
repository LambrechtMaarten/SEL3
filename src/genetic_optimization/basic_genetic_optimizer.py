"""Basic genetic optimizer: top-half selection with Gaussian mutation."""

from typing import Any, Tuple

import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class BasicGeneticOptimizer(GeneticOptimizer):
    """Simple (μ + λ) genetic optimizer with truncation selection and isotropic Gaussian mutation.

    At each generation the top 50 % of individuals (by fitness) are kept and
    cloned with additive N(0, 1) noise to produce an offspring population of
    the same size.
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
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2 :]], None

    def reproduce(self, genomes: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Create offspring by adding isotropic N(0, 1) noise to each selected genome.

        Args:
            genomes: Selected survivor genomes of shape
                ``(n_survivors, genome_size)``.
            evaluations: Unused.
            rng: JAX random key for noise sampling.
            state: Unused.

        Returns:
            Tuple of (new_population, None) where new_population concatenates
            the original survivors with their noisy offspring.
        """
        return jnp.concatenate([genomes, genomes + jax.random.normal(rng, genomes.shape)]), None
