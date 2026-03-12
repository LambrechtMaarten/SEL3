import jax
from jax import numpy as jnp

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class BasicGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        return jnp.concatenate([genomes, genomes + jax.random.normal(rng, genomes.shape)])
