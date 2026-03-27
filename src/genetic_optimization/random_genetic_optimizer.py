from typing import Any, Tuple

import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class RandomNormalGeneticOptimizer(GeneticOptimizer):
    def select(
        self, population: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        return population[
            jnp.argsort(evaluations)[-jnp.size(population, 0) // 2 :]
        ], None

    def reproduce(
        self, genomes: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        rng1, rng2 = jax.random.split(rng)
        multiplier = 0.01 * (
            10 ** jax.random.uniform(rng1, minval=0, maxval=2.4, shape=genomes.shape)
        )
        return jnp.concatenate(
            [genomes, genomes + jax.random.normal(rng2, genomes.shape) * multiplier]
        ), None
