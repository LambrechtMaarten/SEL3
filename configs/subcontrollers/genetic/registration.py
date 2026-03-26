from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from src.genetic_optimization.basic_genetic_optimizer import BasicGeneticOptimizer
from src.genetic_optimization.crossover_genetic_optimizer import (
    CrossoverGeneticOptimizer,
)
from src.genetic_optimization.evosax_genetic_optimizer import (
    EvoSaxGeneticOptimizer,
)
from src.genetic_optimization.random_genetic_optimizer import (
    RandomNormalGeneticOptimizer,
)


def register():
    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration("standard", 10, 10, BasicGeneticOptimizer()),
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration("evosax", 20, 5, EvoSaxGeneticOptimizer()),
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration("crossover", 20, 10, CrossoverGeneticOptimizer()),
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration("short", 20, 3, RandomNormalGeneticOptimizer()),
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration("long", 100, 20, RandomNormalGeneticOptimizer()),
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration("extensive", 200, 50, RandomNormalGeneticOptimizer()),
    )
