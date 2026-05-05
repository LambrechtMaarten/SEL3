import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from configs.subconfigurations.logger.logger import Logger
from src.jax_extra.jax_extra import jarr

evaluator_t = Callable[[jarr], jarr]


class GeneticOptimizer(ABC):
    @staticmethod
    def initialize_population(population_size, genome_size, rng) -> jarr:
        return jax.random.normal(rng, (population_size, genome_size))

    @abstractmethod
    def select(
        self, population: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        pass

    @abstractmethod
    def reproduce(
        self, genomes: jarr, evaluations: jarr, rng, state
    ) -> Tuple[jarr, Any]:
        pass

    def generation(
        self,
        evaluator: evaluator_t,
        population: jarr,
        iterations: int,
        rng: jarr,
        logger: Logger,
    ) -> Tuple[jarr, jarr]:
        evaluations = jnp.zeros(0)
        selections = jnp.zeros(0)
        state = None

        for i in range(iterations):
            t_start = time.time()
            rng, _rng, __rng = jax.random.split(rng, 3)
            if i != 0:
                population, state = self.reproduce(selections, evaluations, _rng, state)
            evaluations = evaluator(population)
            selections, state = self.select(population, evaluations, __rng, state)
            logger.log_genetic_generation(population, selections, evaluations, i)

            # Voortgangsoutput naar terminal
            scores = evaluations[1] if isinstance(evaluations, tuple) else evaluations
            elapsed = time.time() - t_start
            print(
                f"Generatie {i + 1:3d}/{iterations} | "
                f"best: {float(jnp.max(scores)):.4f} | "
                f"gem: {float(jnp.mean(scores)):.4f} | "
                f"tijd: {elapsed:.1f}s"
            )

        return selections, evaluations
