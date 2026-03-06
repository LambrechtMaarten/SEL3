from abc import ABC, abstractmethod

from typing import Callable

import jax
import jax.numpy as jnp

from src.jax_extra import jarr


class GeneticOptimizer(ABC):
    def __init__(self, evaluator: Callable[[jarr], float | jarr], population_size: int, genome_size: int,
                 rng):
        self._evaluator = evaluator
        jax.debug.print("hihi")
        self._population_size = population_size
        self._genome_size = genome_size
        self.rng = rng

    def evaluate_population(self, population: jarr) -> jarr:
        return jax.vmap(self._evaluator)(population)

    def initialize_population(self) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return jax.random.normal(_rng, (self._population_size, self._genome_size))

    @abstractmethod
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        pass

    @abstractmethod
    def reproduce(self, genomes: jarr, evaluations: jarr) -> jarr:
        pass

    def generation(self, population: jarr, iterations: int) -> jarr:
        evaluations = self.evaluate_population(population)
        selections = self.select(population, evaluations)
        for i in range(iterations - 1):
            evaluations = self.evaluate_population(population)
            selections = self.select(population, evaluations)
            population = self.reproduce(selections, evaluations)
            jax.debug.print("{i}: {x}", i=i + 1, x=jnp.max(evaluations))
        return selections


class BasicGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-self._population_size // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return jnp.concatenate([genomes, genomes + jax.random.normal(_rng, genomes.shape) / 20])


class VarianceGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-self._population_size // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return jnp.append(genomes,
                          genomes + jax.random.normal(_rng, genomes.shape) * jnp.sqrt(jnp.abs(jnp.max(evaluations))),
                          axis=0)
