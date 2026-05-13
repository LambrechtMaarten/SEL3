import sys

import jax.numpy as jnp

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import (
    ControllerConfiguration,
)
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.register import register
from configs.subcontrollers.simulation.simulation_configurations import (
    SimulationConfiguration,
)

from src.render.render_video import render_saved_controller
from src.simulation.simulate_controller import simulate_controller
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
            SubConfigurationMap.get_configuration(Logger, "standard"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "evosax"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "standard"),
        )
        train_controller(configuration)

    elif sys.argv[1] == "train_network":
        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "evosax"),
            SubConfigurationMap.get_configuration(
                ControllerConfiguration, "network_multi"
            ),
        )

        # Laad de getrainde CPG-parameters voor RIGHT
        # controller_path = sys.argv[2]
        controller = configuration.controller.controller
        controller.train_controller(configuration)

    elif sys.argv[1] == "train_network_pretrain":
        # Gebruik: python -m src.main train_network_pretrain <pad/naar/expert_gait>
        # De expert-gait is een tekstbestand opgeslagen door OneDirectionController.
        if len(sys.argv) < 3:
            print("Geef het pad naar de expert-gait mee als tweede argument.")
            sys.exit(1)

        with open(sys.argv[2], "r") as f:
            raw = f.read()
        body_cpg = jnp.array(
            [float(x) for x in raw.replace("[", " ").replace("]", " ").split()]
        )

        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "evosax"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "network_pretrain"),
        )

        controller = configuration.controller.controller
        controller.train_controller(configuration, pretrained_body_cpg=body_cpg)

    elif sys.argv[1] == "train_network_archive":
        # Gebruik: python -m src.main train_network_archive <pad/naar/archive/>
        # Het archief bevat selections.npy en x_positions.npy van map-elites.
        # Na pretraining wordt de controller opgeslagen voor evaluatie/render.
        if len(sys.argv) < 3:
            print("Geef het pad naar het map-elites archief mee als tweede argument.")
            sys.exit(1)

        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "evosax"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "network_pretrain"),
        )
        configuration.logger.init_logger()

        controller = configuration.controller.controller
        controller.train_controller(configuration, pretrained_body_cpg=sys.argv[2])
        print(f"Controller opgeslagen in: {configuration.logger.base_folder}")

    elif sys.argv[1] == "simulate":
        simulate_controller(sys.argv[2])

    elif sys.argv[1] == "render":
        render_saved_controller(sys.argv[2])
