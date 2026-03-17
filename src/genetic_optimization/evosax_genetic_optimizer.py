from typing import Any, Tuple

import jax
import jax.numpy as jnp
from evosax.algorithms import PGPE as ES

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class EvoSaxGeneticOptimizer(GeneticOptimizer):
    def __init__(self):
        super().__init__()
        self.es = None
        self.params = None

    def select(
        self, population: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        # Init es for the first time here, as this is not possible at the constructor
        if self.es is None:
            population_size = population.shape[0]
            # ---- genome structure ----
            num_arms = 5
            oscillators_per_arm = 2
            n_osc = num_arms * oscillators_per_arm

            freq_size = 1
            amp_size = n_osc
            off_size = n_osc
            phase_size = n_osc * n_osc
            solution = jnp.concatenate(
                [
                    jnp.ones(amp_size),
                    jnp.ones(freq_size) * 2.0,
                    jnp.linspace(0, 2 * jnp.pi, phase_size),
                    jnp.zeros(off_size),
                ]
            )
            solution += 0.01 * jax.random.normal(rng, solution.shape)

            self.es = ES(
                population_size,
                solution,
            )
            self.params = self.es.default_params.replace(
                std_init=0.1, std_lr=0.05, std_max_change=0.2
            )

            _, subkey = jax.random.split(rng)
            state = self.es.init(subkey, solution, self.params)
            population, state = self.es.ask(rng, state, self.params)
        # Evosax works with fitness
        fitness = -evaluations
        new_state, metrics = self.es.tell(rng, population, fitness, state, self.params)
        return population, new_state

    def reproduce(self, genomes, evaluations, rng, state) -> Tuple[jarr, Any]:
        new_population, state = self.es.ask(rng, state, self.params)

        return new_population, state
