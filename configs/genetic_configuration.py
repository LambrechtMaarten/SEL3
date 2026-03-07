from configs.config import Configuration
from configs.subconfiguration import SubConfiguration
from src.genetic_optimization import GeneticOptimizer, BasicGeneticOptimizer


class GeneticConfiguration(SubConfiguration):
    def __init__(
            self,
            name: str,
            population_size: int,
            generations: int,
            genetic_optimizer: GeneticOptimizer):
        super().__init__(name)
        self.population_size = population_size
        self.generations = generations
        self.genetic_optimizer = genetic_optimizer


standard = lambda: GeneticConfiguration(
    "standard",
    10,
    10,
    BasicGeneticOptimizer()
)
