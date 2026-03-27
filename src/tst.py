import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import imageio
import os
import jax.numpy as jnp

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import ControllerConfiguration
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration


def tst(path: str):
    with open(os.path.join(path, "genetic"), "r") as f:
        arrays = f.readlines()
    jnp_array = jnp.array([float(x) for x in arrays.replace("[", " ").replace("]", " ").split()])
    with open(os.path.join(path, "configuration.json"), "r") as f:
        configuration_json = f.read()
    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "silent_logger"),
        SubConfigurationMap.get_configuration_from_json(configuration_json, SimulationConfiguration),
        SubConfigurationMap.get_configuration_from_json(configuration_json, CPGConfiguration),
        SubConfigurationMap.get_configuration_from_json(configuration_json, RandomConfiguration),
        SubConfigurationMap.get_configuration_from_json(configuration_json, GeneticConfiguration),
        SubConfigurationMap.get_configuration_from_json(configuration_json, ControllerConfiguration)
    )
    genome_size = configuration.controller.controller.genome_size(configuration)
    print(genome_size)
    print(len(jnp_array))
    exit(0)
    populations = [np.random.rand(100, 67) for _ in range(20)]
    selections = [pop[:50] for pop in populations]

    generations = len(populations)

    # -------------------------
    # COLLECT ALL GENOMES
    # -------------------------
    all_genomes = np.vstack(populations)

    print("Running t-SNE on", all_genomes.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )

    embedding = tsne.fit_transform(all_genomes)

    # split embedding back into generations
    embeddings_per_gen = []
    start = 0
    for pop in populations:
        end = start + len(pop)
        embeddings_per_gen.append(embedding[start:end])
        start = end

    # -------------------------
    # CREATE OUTPUT FOLDER
    # -------------------------
    frame_dir = "tsne_frames"
    os.makedirs(frame_dir, exist_ok=True)

    frames = []

    # -------------------------
    # PLOT EACH GENERATION
    # -------------------------
    for gen in range(generations):

        pop = populations[gen]
        sel = selections[gen]
        emb = embeddings_per_gen[gen]

        # determine which population members are selected
        selected_mask = np.zeros(len(pop), dtype=bool)

        for s in sel:
            matches = np.where((pop == s).all(axis=1))[0]
            if len(matches) > 0:
                selected_mask[matches[0]] = True

        pop_points = emb[~selected_mask]
        sel_points = emb[selected_mask]

        plt.figure(figsize=(6, 6))

        plt.scatter(pop_points[:, 0], pop_points[:, 1],
                    c="blue", s=40, label="Population")

        plt.scatter(sel_points[:, 0], sel_points[:, 1],
                    c="yellow", edgecolors="black", s=70, label="Selected")

        plt.title(f"Generation {gen}")
        plt.legend()
        plt.tight_layout()

        frame_path = f"{frame_dir}/frame_{gen:03d}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))

    # -------------------------
    # CREATE GIF
    # -------------------------
    gif_path = "genetic_algorithm_tsne.gif"
    imageio.mimsave(gif_path, frames, duration=10)

    print("GIF saved to:", gif_path)


if __name__ == '__main__':
    if sys.argv[1] == "test":
        tst(sys.argv[2])
