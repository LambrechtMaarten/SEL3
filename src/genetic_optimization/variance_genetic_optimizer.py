from typing import Any, Tuple

import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class VarianceGeneticOptimizer(GeneticOptimizer):
    def select(
        self, population: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        return population[
            jnp.argsort(evaluations)[-jnp.size(population, 0) // 2 :]
        ], None

    def reproduce(
        self, genomes: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        return jnp.concatenate(
            [
                genomes,
                genomes
                + jax.random.normal(rng, genomes.shape)
                * jnp.sqrt(jnp.abs(jnp.max(evaluations))),
            ]
        ), None
