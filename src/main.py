import sys

import jax.debug
import jax.numpy as jnp

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subconfigurations.controller.controller_configurations import (
    ControllerConfiguration,
)
from configs.subconfigurations.cpg.cpg_configurations import CPGConfiguration
from configs.subconfigurations.genetic.genetic_configurations import GeneticConfiguration
from configs.subconfigurations.logger.logger import Logger
from configs.subconfigurations.random.random_configurations import RandomConfiguration
from configs.subconfigurations.register import register
from configs.subconfigurations.simulation.simulation_configurations import (
    SimulationConfiguration,
)
from src.controller.network_controller import NetworkController
from src.render.render_video import render_saved_controller
from src.simulation.simulate_controller import simulate_controller
from src.training.train_archive import train_archive
from src.training.train_controller import train_controller

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'you need to say "train", "train_network", "simulate" or "render" '
            "as the first arg or it doesn't know what to do"
        )

    register()

    if sys.argv[1] == "train":
        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "short"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "standard"),
        )
        train_controller(configuration)

    if sys.argv[1] == "map-elites":
        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "standard"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "short"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "map elites"),
        )
        groups, edges = train_archive(configuration)
        jax.debug.log("{x}",x=edges)
        jax.debug.log("{x}",x=groups)

    elif sys.argv[1] == "train_network":
        if len(sys.argv) < 3:
            print("Geef het pad naar de getrainde CPG controller mee als argument")
            sys.exit(1)

        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "map elites"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "map elites"),
        )

        # Laad de getrainde CPG-parameters voor RIGHT
        controller_path = sys.argv[2]
        with open(controller_path, "r") as f:
            arrays = f.read()
            cpg_params_right = jnp.array(
                [float(x) for x in arrays.replace("[", " ").replace("]", " ").split()]
            )

        # Fase 1: pre-train op 5 natuurlijke richtingen
        network_controller: NetworkController = configuration.controller.controller
        network_controller.pretrain_from_cpg(
            cpg_params_right, configuration, learning_rate=0.01, steps=1000
        )

        # Fase 2: genetische optimalisatie voor interpolatie
        train_controller(configuration)

    elif sys.argv[1] == "simulate":
        simulate_controller(sys.argv[2])

    elif sys.argv[1] == "render":
        render_saved_controller(sys.argv[2])
