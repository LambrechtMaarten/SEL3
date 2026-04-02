import os
import sys

import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

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


def tst(path: str):
    register()
    with open(os.path.join(path, "genetic"), "r") as f:
        arrays = f.read()
    values = [float(x) for x in arrays.replace("[", " ").replace("]", " ").split()]
    with open(os.path.join(path, "configuration.json"), "r") as f:
        configuration_json = f.read()
    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "silent_logger"),
        SubConfigurationMap.get_configuration_from_json(
            configuration_json, SimulationConfiguration
        ),
        SubConfigurationMap.get_configuration_from_json(
            configuration_json, CPGConfiguration
        ),
        SubConfigurationMap.get_configuration_from_json(
            configuration_json, RandomConfiguration
        ),
        SubConfigurationMap.get_configuration_from_json(
            configuration_json, GeneticConfiguration
        ),
        SubConfigurationMap.get_configuration_from_json(
            configuration_json, ControllerConfiguration
        ),
    )
    genome_size = configuration.controller.controller.genome_size(configuration)
    population_size = configuration.genetic.population_size
    generations = configuration.genetic.number_of_generations
    populations = []
    selections = []
    evaluations = []
    i = 0
    for _ in range(generations):
        for _ in range(genome_size * population_size):
            populations.append(values[i])
            i += 1
        for _ in range(population_size):
            evaluations.append(values[i])
            i += 1
        for _ in range(genome_size * population_size // 2):
            selections.append(values[i])
            i += 1

    populations = jnp.array(populations).reshape(
        generations, population_size, genome_size
    )
    selections = jnp.array(selections).reshape(
        generations, population_size // 2, genome_size
    )
    evaluations = jnp.array(evaluations).reshape(generations, population_size)

    all_genomes = jnp.vstack(populations)

    embedding = TSNE(
        n_components=2,
        perplexity=100,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(all_genomes)

    embeddings_per_gen = []
    start = 0
    for pop in populations:
        end = start + len(pop)
        embeddings_per_gen.append(embedding[start:end])
        start = end
    print(embeddings_per_gen)
    min_coord: float = jnp.min(jnp.array(embeddings_per_gen))
    max_coord: float = jnp.max(jnp.array(embeddings_per_gen))
    print(min_coord, max_coord)

    frame_dir = "tsne_frames"
    os.makedirs(frame_dir, exist_ok=True)

    frames = []

    for gen in range(generations):
        pop = populations[gen]
        sel = selections[gen]
        emb = embeddings_per_gen[gen]

        selected_mask = np.zeros(len(pop), dtype=bool)

        for s in sel:
            matches = np.where((pop == s).all(axis=1))[0]
            if len(matches) > 0:
                selected_mask[matches[0]] = True

        pop_points = emb[~selected_mask]
        sel_points = emb[selected_mask]

        plt.figure(figsize=(6, 6))
        plt.gca().set_xlim(min_coord, max_coord)
        plt.gca().set_ylim(min_coord, max_coord)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.scatter(
            pop_points[:, 0], pop_points[:, 1], c="blue", s=40, label="Population"
        )

        plt.scatter(
            sel_points[:, 0],
            sel_points[:, 1],
            c="yellow",
            edgecolors="black",
            s=70,
            label="Selected",
        )

        plt.title(f"Generation {gen}")
        plt.legend()
        plt.tight_layout()

        frame_path = f"{frame_dir}/frame_{gen:03d}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = "genetic_algorithm_tsne.gif"
    imageio.mimsave(gif_path, frames, fps=10, loop=0)

    print("GIF saved to:", gif_path)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        tst(sys.argv[2])
