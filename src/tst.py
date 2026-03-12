import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import imageio
import os

# -------------------------
# INPUT DATA
# -------------------------
# populations: list of 20 arrays of shape (100, 67)
# selections: list of 20 arrays of shape (50, 67)

# Example placeholders (REMOVE if you already have real data)
populations = [np.random.rand(100,67) for _ in range(20)]
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

    plt.figure(figsize=(6,6))

    plt.scatter(pop_points[:,0], pop_points[:,1],
                c="blue", s=40, label="Population")

    plt.scatter(sel_points[:,0], sel_points[:,1],
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