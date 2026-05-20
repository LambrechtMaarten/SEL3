import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_arrays_from_file(filename):
    with open(filename, "r") as f:
        content = f.read()

    blocks = re.findall(r"\[.*?\]", content, re.DOTALL)

    arrays = []
    for block in blocks:
        numbers = list(map(float, re.findall(r"-?\d+\.?\d*e?\+?\d*", block)))
        arrays.append(np.array(numbers))

    pairs = [(arrays[i], arrays[i + 1]) for i in range(0, len(arrays), 2)]
    return pairs


def reduce_to_best(x, y):
    """Return sorted unique x with max y per x"""
    best = defaultdict(lambda: float("-inf"))

    for xi, yi in zip(x, y):
        if yi > best[xi]:
            best[xi] = yi

    x_sorted = sorted(best.keys())
    y_best = [best[xi] for xi in x_sorted]

    return np.array(x_sorted), np.clip(np.log(np.array(y_best)), 0, 12)


def plot_pairs(pairs):
    n = len(pairs)
    plt.figure(figsize=(8, 5))

    for i, (x, y) in enumerate(pairs):
        x_best, y_best = reduce_to_best(x, y)

        # Color interpolation: blue → red
        t = i / (n - 1) if n > 1 else 0
        color = (t, 0, 1 - t)

        i = i + 1
        plt.plot(
            x_best,
            y_best,
            marker="o",
            color=color,
            label=i if i % 10 == 0 else ("..." if i % 10 == 6 else None),
        )

    plt.title(f"map elites: {len(pairs)} generations")
    plt.xlabel("speed")
    plt.ylabel("efficiency (logarithmic scale)")
    plt.grid(True)
    plt.legend()

    plt.show()


def plot_pairs2(pairs):
    n = len(pairs)
    plt.figure(figsize=(8, 5))

    for i, (x, y) in enumerate(pairs):
        x_best, y_best = reduce_to_best(x, y)

        # Color interpolation: blue → red
        t = i / (n - 1) if n > 1 else 0
        color = (t, 0, 1 - t)

        i = i + 1
        plt.scatter(
            x_best,
            y_best,
            marker="o",
            color=color,
            alpha=1,
            label=i if i % 10 == 0 else ("..." if i % 10 == 6 else None),
        )

    plt.title(f"map elites: {len(pairs)} generations")
    plt.xlabel("speed")
    plt.ylabel("efficiency (logarithmic scale)")
    plt.grid(True)
    plt.legend()

    plt.show()


pairs = parse_arrays_from_file("tmp4")
plot_pairs(pairs)
plot_pairs2(pairs)
