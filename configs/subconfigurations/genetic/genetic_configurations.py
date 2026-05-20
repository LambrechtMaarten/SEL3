from configs.subconfiguration import SubConfiguration
from src.genetic_optimization.genetic_optimization import GeneticOptimizer


class GeneticConfiguration(SubConfiguration):
    """
    This class contains the configuration of the genetic algorithm.
    It determines the population size, number of generations, and algorithm used for learning.
    """

    def __init__(
        self, name: str, population_size: int, generations: int, genetic_optimizer: GeneticOptimizer
    ):
        super().__init__(name)
        self.population_size = population_size
        self.number_of_generations = generations
        self.genetic_optimizer = genetic_optimizer
