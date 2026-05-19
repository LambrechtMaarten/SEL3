import os
import time
from pathlib import Path

import jax
import numpy as np
from fontTools.ufoLib import convertFontInfoValueForAttributeFromVersion2ToVersion1

from configs.config import Configuration
from configs.subconfigurations.controller.controller_configurations import ControllerConfiguration
from src.controller.control_input import ControlInput
from src.controller.one_direction_map_elites_controller import OneDirectionMapElitesController
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr


def train_archive(configuration: Configuration):
    configuration.controller = ControllerConfiguration("map elites", OneDirectionMapElitesController())
    configuration.controller.set_configuration(configuration)

    configuration.logger.init_logger()
    configuration.logger.log_configuration()

    genetic_optimizer = configuration.genetic.genetic_optimizer
    controller = configuration.controller.controller

    starting_population = genetic_optimizer.initialize_population(
        configuration.genetic.population_size,
        controller.genome_size(configuration),
        configuration.random.split(),
    )

    evaluator_fn = jax.jit(controller.evaluator(configuration, configuration.random.split()))
    selections, evaluations = genetic_optimizer.generation(
        evaluator_fn,
        starting_population,
        configuration.genetic.number_of_generations,
        configuration.random.split(),
        configuration.logger,
    )

    controller: OneDirectionMapElitesController = configuration.controller.controller
    get_edges = jax.jit(controller.get_edges(configuration, configuration.random.split()))
    groups, edges = get_edges(selections)

    # Sla het archief op zodat het gebruikt kan worden voor pretraining
    archive_dir = Path(configuration.logger.base_folder) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    np.save(archive_dir / "groups.npy", np.array(groups))
    np.save(archive_dir / "edges.npy", np.array(edges))
    np.save(archive_dir / "selections.npy", np.array(selections))
    np.save(archive_dir / "evaluations.npy", np.array(evaluations[1] if isinstance(evaluations, tuple) else evaluations))

    print(f"Archief opgeslagen in: {archive_dir}")
    print(f"  groups.npy     — shape {np.array(groups).shape}")
    print(f"  edges.npy      — shape {np.array(edges).shape}")
    print(f"  selections.npy — shape {np.array(selections).shape}")

    return groups, edges
