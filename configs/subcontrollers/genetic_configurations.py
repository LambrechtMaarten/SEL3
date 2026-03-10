from configs.subconfiguration import SubConfiguration
from src.genetic_optimization import GeneticOptimizer, BasicGeneticOptimizer, RandomNormalGeneticOptimizer


class GeneticConfiguration(SubConfiguration):
    """
    This class contains the configuration of the genetic algorithm.
    It determines the population size, number of generations, and algorithm used for learning.
    """

    def __init__(
            self,
            name: str,
            population_size: int,
            generations: int,
            genetic_optimizer: GeneticOptimizer):
        super().__init__(name)
        self.population_size = population_size
        self.number_of_generations = generations
        self.genetic_optimizer = genetic_optimizer


standard = lambda: GeneticConfiguration(
    "standard",
    10,
    10,
    BasicGeneticOptimizer()
)

short = lambda: GeneticConfiguration(
    "short",
    20,
    3,
    RandomNormalGeneticOptimizer()
)

long = lambda: GeneticConfiguration(
    "long",
    100,
    20,
    RandomNormalGeneticOptimizer()
)

extensive = lambda: GeneticConfiguration(
    "extensive",
    200,
    50,
    RandomNormalGeneticOptimizer()
)
