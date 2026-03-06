from abc import ABC, abstractmethod

from typing import Callable

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

from src.jax_extra import jarr

# def visualize():
#     def evaluator(arr):
#         return -jnp.sum(jnp.square(arr-50))
#
#     genetic_optimizer = VarianceGeneticOptimizer(evaluator, 100, 2)
#
#     points = []
#     population = genetic_optimizer.initialize_population()
#     points.append(population)
#     for i in range(100):
#         evaluations = genetic_optimizer.evaluate_population(population)
#         selections = genetic_optimizer.select(population, evaluations)
#         print(evaluations[0])
#         population = genetic_optimizer.reproduce(selections, evaluations)
#         points.append(population)
#
#
#     points = jnp.array(points)  # shape: (frames, num_points, 2)
#
#     fig, ax = plt.subplots()
#
#     scat = ax.scatter([], [], s=.4)
#
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0, 100)
#
#     def update(frame):
#         xy = points[frame]
#         scat.set_offsets(xy)
#         return scat,
#
#     ani = FuncAnimation(
#         fig,
#         update,
#         frames=len(points),
#         interval=200,
#         blit=True
#     )
#
#     # Save GIF
#     ani.save("points_animation2.gif", writer=PillowWriter(fps=20))


class GeneticOptimizer(ABC):
    def __init__(self, evaluator: Callable[[jarr], float | jarr], population_size: int, genome_size: int,
                 rng):
        self._evaluator = evaluator
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
        for i in range(iterations-1):
            evaluations = self.evaluate_population(population)
            selections = self.select(population, evaluations)
            population = self.reproduce(selections, evaluations)
            jax.debug.print("{i}: {x}", i=i+1, x=jnp.max(evaluations))
        return selections


class BasicGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-self._population_size // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return jnp.append(genomes, genomes + jax.random.normal(_rng, genomes.shape)/2, axis=0)

class VarianceGeneticOptimizer(GeneticOptimizer):
    def select(self, population: jarr, evaluations: jarr) -> jarr:
        return population[jnp.argsort(evaluations)[-self._population_size // 2:]]

    def reproduce(self, genomes: jarr, evaluations: jarr) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return jnp.append(genomes, genomes + jax.random.normal(_rng, genomes.shape) * jnp.sqrt(jnp.abs(jnp.max(evaluations))), axis=0)
