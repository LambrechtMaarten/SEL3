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

from src.render.render_video import render_saved_controller
from src.simulation.simulate_controller import simulate_controller
from src.training.train_archive import train_archive
from src.training.train_controller import train_controller
from src.simulation.robot_controller import simulate_controller_joystick

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
            SubConfigurationMap.get_configuration(GeneticConfiguration, "map elites"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "map elites"),
        )
        groups, edges = train_archive(configuration)
        jax.debug.log("{x}",x=edges)
        jax.debug.log("{x}",x=groups)

    elif sys.argv[1] == "train_network":
        configuration = Configuration(
            SubConfigurationMap.get_configuration(Logger, "wandb"),
            SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
            SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
            SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
            SubConfigurationMap.get_configuration(GeneticConfiguration, "map elites"),
            SubConfigurationMap.get_configuration(ControllerConfiguration, "map elites"),
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

    elif sys.argv[1] == "simulate_controller":
        simulate_controller_joystick(sys.argv[2])

    elif sys.argv[1] == "render":
        render_saved_controller(sys.argv[2])
