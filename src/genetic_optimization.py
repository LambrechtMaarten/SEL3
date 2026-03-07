from abc import ABC, abstractmethod

from typing import Callable

import jax
import jax.numpy as jnp

from src.cpg.cpg import CPG
from src.env import Environment
from src.jax_extra import jarr

evaluator_t = Callable[[jarr, Environment, CPG], jarr]


class GeneticOptimizer(ABC):
    @staticmethod
    def initialize_population(population_size, genome_size, rng) -> jarr:
        return jax.random.normal(rng, (population_size, genome_size))

    @abstractmethod
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        pass

    @abstractmethod
    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        pass

    def generation(
            self, evaluator: evaluator_t,
            population: jarr,
            iterations: int,
            rng: jarr,
            env: Environment,
            cpg: CPG) -> jarr:
        evaluations = evaluator(population, env, cpg)
        selections = self.select(population, evaluations)
        for i in range(iterations - 1):
            rng, _rng = jax.random.split(rng)
            evaluations = evaluator(population, env, cpg)
            selections = self.select(population, evaluations)
            population = self.reproduce(selections, evaluations, _rng)
            jax.debug.print("{i}: {x}", i=i + 1, x=jnp.max(evaluations))
        return selections


class BasicGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        return jnp.concatenate([genomes, genomes + jax.random.normal(rng, genomes.shape) / 20])


class VarianceGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        return jnp.concatenate([
            genomes,
            genomes + jax.random.normal(rng, genomes.shape) * jnp.sqrt(jnp.abs(jnp.max(evaluations)))
        ])
