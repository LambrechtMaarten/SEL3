"""Genetic optimizer that scales mutation noise by the square root of the best fitness."""

from typing import Any, Tuple

import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class VarianceGeneticOptimizer(GeneticOptimizer):
    """Genetic optimizer with fitness-adaptive mutation variance.

    Selection keeps the top 50 % of individuals.  The mutation step size is
    scaled by sqrt(|max_fitness|), so that the optimizer takes larger steps
    when fitness values are large and smaller steps when near convergence.
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
        """Create offspring using mutation scaled by the maximum fitness magnitude.

        The standard deviation of the Gaussian noise is
        ``sqrt(|max(evaluations)|)``.

        Args:
            genomes: Selected survivor genomes of shape
                ``(n_survivors, genome_size)``.
            evaluations: Fitness scores used to compute the mutation scale.
            rng: JAX random key.
            state: Unused.

        Returns:
            Tuple of (new_population, None) concatenating survivors and
            offspring.
        """
        return jnp.concatenate(
            [
                genomes,
                genomes
                + jax.random.normal(rng, genomes.shape) * jnp.sqrt(jnp.abs(jnp.max(evaluations))),
            ]
        ), None
