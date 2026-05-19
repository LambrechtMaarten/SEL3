import sys

import jax.debug

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
from src.render.render_video import render_saved_controller
from src.simulation.robot_controller import simulate_controller_joystick
from src.simulation.simulate_controller import simulate_controller
from src.training.train_archive import train_archive

if __name__ == "__main__":
    valid_commands = {
        "map-elites",
        "train_network",
        "train_network_pretrain",
        "simulate",
        "simulate_controller",
        "render",
    }

    if len(sys.argv) < 2:
        print("Error: geen commando opgegeven.")
        print(f"Gebruik één van: {', '.join(valid_commands)}")
        sys.exit(1)

    if sys.argv[1] not in valid_commands:
        print(f'Error: onbekend commando "{sys.argv[1]}"')
        print(f"Gebruik één van: {', '.join(valid_commands)}")
        sys.exit(1)

    register()

    if sys.argv[1] == "map-elites":
        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "standard"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "map elites"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "map elites"),
        )
        groups, edges = train_archive(configuration)
        jax.debug.log("{x}", x=edges)
        jax.debug.log("{x}", x=groups)

    elif sys.argv[1] == "train_network":
        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "symmetric"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "map elites"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "network"),
        )

        controller = configuration.controller.controller
        controller.train_controller(configuration)
        print(f"Controller opgeslagen in: {configuration.logger.base_folder}")
    elif sys.argv[1] == "train_network_pretrain":
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
            SubConfigurationMap.get_configuration(GeneticConfiguration, "map elites"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "network_pretrain"),
        )

        controller = configuration.controller.controller
        controller.train_controller(configuration, archive=sys.argv[2])
        print(f"Controller opgeslagen in: {configuration.logger.base_folder}")

    elif sys.argv[1] == "simulate":
        simulate_controller(sys.argv[2])

    elif sys.argv[1] == "simulate_controller":
        simulate_controller_joystick(sys.argv[2])

    elif sys.argv[1] == "render":
        render_saved_controller(sys.argv[2])
