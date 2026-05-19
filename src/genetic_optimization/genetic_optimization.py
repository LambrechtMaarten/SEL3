"""Abstract base class for genetic / evolutionary optimizers."""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from configs.subconfigurations.logger.logger import Logger
from src.jax_extra.jax_extra import jarr

evaluator_t = Callable[[jarr], jarr]
"""Type alias for a fitness evaluator: maps a genome batch to fitness scores."""


class GeneticOptimizer(ABC):
    """Abstract base class for genetic / evolutionary optimizers.

    Subclasses must implement ``select`` (survival selection) and
    ``reproduce`` (offspring generation).  The ``generation`` method
    orchestrates the full optimisation loop.
    """

    @staticmethod
    def initialize_population(population_size, genome_size, rng) -> jarr:
        """Draw an initial population from a standard normal distribution.

        Args:
            population_size: Number of individuals in the population.
            genome_size: Number of genes (parameters) per individual.
            rng: JAX random key.

        Returns:
            JAX array of shape ``(population_size, genome_size)`` sampled
            from N(0, 1).
        """
        return jax.random.normal(rng, (population_size, genome_size))

    @abstractmethod
    def select(self, population: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Select survivors from the current population.

        Args:
            population: Current genome array of shape
                ``(population_size, genome_size)``.
            evaluations: Fitness scores of shape ``(population_size,)`` or a
                tuple of score arrays.
            rng: JAX random key.
            state: Optimizer-specific state from the previous generation, or
                None for the first generation.

        Returns:
            Tuple of (selected_genomes, new_state).
        """

    @abstractmethod
    def reproduce(self, genomes: jarr, evaluations: jarr, rng, state) -> Tuple[jarr, Any]:
        """Generate offspring from the selected survivors.

        Args:
            genomes: Selected survivor genomes of shape
                ``(n_survivors, genome_size)``.
            evaluations: Fitness scores corresponding to ``genomes``.
            rng: JAX random key.
            state: Optimizer-specific state from ``select``.

        Returns:
            Tuple of (new_population, new_state) to be evaluated in the next
            generation.
        """

    def generation(
        self,
        evaluator: evaluator_t,
        population: jarr,
        iterations: int,
        rng: jarr,
        logger: Logger,
    ) -> Tuple[jarr, jarr]:
        """Run the full genetic optimisation loop for a fixed number of generations.

        Each iteration: (1) reproduce from previous survivors (skipped for
        generation 0), (2) evaluate the population, (3) select survivors, and
        (4) log progress.

        Args:
            evaluator: Fitness function mapping a genome batch to scores.
            population: Initial genome array of shape
                ``(population_size, genome_size)``.
            iterations: Number of generations to run.
            rng: JAX random key.
            logger: Logger instance for per-generation logging.

        Returns:
            Tuple of (final_selections, final_evaluations).
        """
        evaluations = jnp.zeros(0)
        selections = jnp.zeros(0)
        state = None

        for i in range(iterations):
            t_start = time.time()
            rng, _rng, __rng = jax.random.split(rng, 3)
            if i != 0:
                print(f"Generation {i + 1:3d}/{iterations} | reproducing...", flush=True)
                population, state = self.reproduce(selections, evaluations, _rng, state)
            print(
                f"Generation {i + 1:3d}/{iterations} | evaluating ({len(population)} genomes)...",
                flush=True,
            )
            evaluations = evaluator(population)
            selections, state = self.select(population, evaluations, __rng, state)
            logger.log_genetic_generation(population, selections, evaluations, i)

            # Progress output to terminal
            scores = evaluations[1] if isinstance(evaluations, tuple) else evaluations
            elapsed = time.time() - t_start
            print(
                f"Generation {i + 1:3d}/{iterations} | "
                f"best: {float(jnp.max(scores)):.4f} | "
                f"mean: {float(jnp.mean(scores)):.4f} | "
                f"time: {elapsed:.1f}s",
                flush=True,
            )

        return selections, evaluations
