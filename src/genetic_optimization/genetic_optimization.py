from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from configs.subcontrollers.logger.logger import Logger
from src.jax_extra.jax_extra import jarr

evaluator_t = Callable[[jarr], jarr]


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
            self,
            evaluator: evaluator_t,
            population: jarr,
            iterations: int,
            rng: jarr,
            logger: Logger) -> Tuple[jarr, jarr]:
        evaluations = jnp.zeros(0)
        selections = jnp.zeros(0)

        for i in range(iterations):
            rng, _rng = jax.random.split(rng)
            if i != 0:
                population = self.reproduce(selections, evaluations, _rng)
            evaluations = evaluator(population)
            selections = self.select(population, evaluations)
            logger.log_genetic_generation(population, evaluations, selections)
            logger.log(f'{i}:\t{jnp.max(evaluations)}')

        return selections, evaluations


class BasicGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        return jnp.concatenate([genomes, genomes + jax.random.normal(rng, genomes.shape)])


class RandomNormalGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        rng1, rng2 = jax.random.split(rng)
        multiplier = 0.01 * (10 ** jax.random.uniform(rng1, minval=0, maxval=2.4, shape=genomes.shape))
        return jnp.concatenate([genomes, genomes + jax.random.normal(rng2, genomes.shape) * multiplier])


class VarianceGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-jnp.size(population, 0) // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr, rng) -> jarr:
        return jnp.concatenate([
            genomes,
            genomes + jax.random.normal(rng, genomes.shape) * jnp.sqrt(jnp.abs(jnp.max(evaluations)))
        ])
