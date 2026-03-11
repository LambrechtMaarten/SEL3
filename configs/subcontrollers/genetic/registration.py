from configs.subconfiguration import SubConfigurationMap
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from src.genetic_optimization import BasicGeneticOptimizer, RandomNormalGeneticOptimizer


def register():
    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration(
            "standard",
            10,
            10,
            BasicGeneticOptimizer()
        )
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration(
            "short",
            20,
            3,
            RandomNormalGeneticOptimizer()
        )
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration(
            "long",
            100,
            20,
            RandomNormalGeneticOptimizer()
        )
    )

    SubConfigurationMap.add_configuration(
        GeneticConfiguration,
        GeneticConfiguration(
            "extensive",
            200,
            50,
            RandomNormalGeneticOptimizer()
        )
    )
