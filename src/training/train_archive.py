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
    x_positions, groups, edges = get_edges(selections)

    # Sla het archief op zodat het gebruikt kan worden voor pretraining
    archive_dir = Path(configuration.logger.base_folder) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    np.save(archive_dir / "x_positions.npy", np.array(x_positions))
    np.save(archive_dir / "edges.npy", np.array(edges))
    np.save(archive_dir / "selections.npy", np.array(selections))
    np.save(archive_dir / "evaluations.npy", np.array(evaluations[1] if isinstance(evaluations, tuple) else evaluations))

    print(f"Archief opgeslagen in: {archive_dir}")
    print(f"  x_positions.npy     — shape {np.array(x_positions).shape}")
    print(f"  edges.npy      — shape {np.array(edges).shape}")
    print(f"  selections.npy — shape {np.array(selections).shape}")

    seen_groups = []
    for i in range(len(groups)):
        if groups[i] in seen_groups:
            continue
        env = Environment(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        env_state = env.reset(configuration.random.split())
        cpg_state = cpg.reset()
        frames = []
        for _ in range(2000):
            cpg_state = configuration.cpg.cpg_generator.modulate_body(cpg_state, selections[i])
            cpg_state = cpg.step(cpg_state)
            actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions, env_state)
            frames.append(env.render(env_state))
        configuration.logger.log_video(frames, f'video_{groups[i]}.mp4')
        seen_groups.append(groups[i])

    return x_positions, edges
