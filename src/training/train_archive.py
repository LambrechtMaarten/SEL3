"""Training script that builds a map-elites CPG gait archive."""

from pathlib import Path

import jax
import numpy as np

from configs.config import Configuration
from configs.subconfigurations.controller.controller_configurations import ControllerConfiguration
from src.controller.one_direction_map_elites_controller import OneDirectionMapElitesController
from src.environment.environment import Environment


def train_archive(configuration: Configuration):
    """Run the map-elites genetic optimisation and save the resulting gait archive.

    Overrides the controller in the provided configuration with a
    :class:`OneDirectionMapElitesController`, runs the genetic optimizer for
    the configured number of generations, and saves the archive artefacts
    (groups, joint-position edges, genome selections, and fitness evaluations)
    to ``<logger.base_folder>/archive/``.

    Args:
        configuration: Global simulation and training configuration.  The
            controller sub-configuration is overwritten internally.

    Returns:
        Tuple of (groups, edges) where ``groups`` is an integer array of
        behavioural bin indices and ``edges`` is a joint-position trajectory
        array of shape ``(N, 800, 30)``.
    """
    controller = OneDirectionMapElitesController()
    configuration.controller = ControllerConfiguration("map elites", controller)
    configuration.controller.set_configuration(configuration)

    configuration.logger.init_logger()
    configuration.logger.log_configuration()

    genetic_optimizer = configuration.genetic.genetic_optimizer

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

    get_edges = jax.jit(controller.get_edges(configuration, configuration.random.split()))
    x_positions, groups, edges = get_edges(selections)

    # Save the archive so it can be used for pre-training
    archive_dir = Path(configuration.logger.base_folder) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    np.save(archive_dir / "x_positions.npy", np.array(x_positions))
    np.save(archive_dir / "edges.npy", np.array(edges))
    np.save(archive_dir / "selections.npy", np.array(selections))
    np.save(
        archive_dir / "evaluations.npy",
        np.array(evaluations[1] if isinstance(evaluations, tuple) else evaluations),
    )

    print(f"Archive saved to: {archive_dir}")
    print(f"  x_positions.npy     — shape {np.array(x_positions).shape}")
    print(f"  edges.npy      — shape {np.array(edges).shape}")
    print(f"  selections.npy — shape {np.array(selections).shape}")

    # create a video of a brittle start for every group in the archive
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
        configuration.logger.log_video(frames, f"video_{groups[i]}.mp4")
        seen_groups.append(groups[i])

    return x_positions, edges
