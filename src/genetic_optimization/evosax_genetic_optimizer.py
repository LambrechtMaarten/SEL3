from typing import Any, Tuple

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES as ES

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class EvoSaxGeneticOptimizer(GeneticOptimizer):
    def __init__(self):
        super().__init__()
        self.es = None

    def select(
        self, population: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        # Init es for the first time here, as this is not possible at the constructor
        if self.es is None:
            population_size = population.shape[0]
            solution = jnp.mean(population, axis=0)
            self.es = ES(population_size, solution)
            _, subkey = jax.random.split(rng)
            state = self.es.init(subkey, solution, self.es.default_params)
        # Evosax works with fitness
        fitness = -evaluations
        new_state, metrics = self.es.tell(
            rng, population, fitness, state, self.es.default_params
        )
        return population, new_state

    def reproduce(self, genomes, evaluations, rng, state) -> Tuple[jarr, Any]:
        new_population, state = self.es.ask(rng, state, self.es.default_params)
        return new_population, state
