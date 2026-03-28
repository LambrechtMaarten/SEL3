from typing import Any, Tuple

import jax
import jax.numpy as jnp
from evosax.algorithms import PGPE as ES

from src.genetic_optimization.genetic_optimization import GeneticOptimizer
from src.jax_extra.jax_extra import jarr


class EvoSaxGeneticOptimizer(GeneticOptimizer):
    """Genetic optimizer using the PGPE algorithm from EvoSax.

    PGPE (Policy Gradients with Parameter-based Exploration) is a
    gradient-free evolutionary strategy that maintains a distribution
    over parameters and updates it based on fitness evaluations.

    The EvoSax strategy is initialized lazily on the first call to
    select(), as the genome size is not known at construction time.
    The genome size is determined dynamically from the population shape,
    making this optimizer compatible with any controller.

    Attributes:
        es: The EvoSax PGPE strategy instance, initialized on first call.
        params: The EvoSax strategy hyperparameters.
    """

    def __init__(self):
        """Initializes the optimizer with no EvoSax strategy yet."""
        super().__init__()
        self.es = None
        self.params = None

    def select(
        self, population: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        """Updates the EvoSax strategy with fitness evaluations.

        Initializes the EvoSax strategy on the first call using the
        genome size derived dynamically from the population shape.
        On subsequent calls, updates the strategy distribution using
        the current population and fitness scores.

        Args:
            population: JAX array of shape (population_size, genome_size).
            evaluations: JAX array of shape (population_size,) with fitness scores.
            rng: A JAX random key for strategy initialization and updates.
            state: EvoSax strategy state from the previous generation,
                or None on the first call.

        Returns:
            Tuple of (population, new_state) where new_state is the
            updated EvoSax strategy state.
        """
        if self.es is None:
            population_size = population.shape[0]
            genome_size = population.shape[1]  # dynamisch, geen hardcoded structuur

            _, subkey = jax.random.split(rng)
            solution = jax.random.normal(subkey, (genome_size,)) * 0.01

            self.es = ES(population_size, solution)
            self.params = self.es.default_params.replace(
                std_init=0.1, std_lr=0.05, std_max_change=0.2
            )

            _, subkey = jax.random.split(rng)
            state = self.es.init(subkey, solution, self.params)
            population, state = self.es.ask(rng, state, self.params)

        fitness = -evaluations
        new_state, metrics = self.es.tell(rng, population, fitness, state, self.params)
        return population, new_state

    def reproduce(
        self, genomes: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        """Samples a new population from the updated EvoSax strategy.

        Args:
            genomes: JAX array of selected genomes (unused, EvoSax
                manages its own population internally).
            evaluations: JAX array of fitness scores (unused).
            rng: A JAX random key for sampling.
            state: EvoSax strategy state from the previous call to select.

        Returns:
            Tuple of (new_population, new_state) sampled from the
            current strategy distribution.
        """
        new_population, state = self.es.ask(rng, state, self.params)
        return new_population, state