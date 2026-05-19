import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_arrays_from_file(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Extract all [ ... ] blocks
    blocks = re.findall(r"\[.*?\]", content, re.DOTALL)

    arrays = []
    for block in blocks:
        numbers = list(map(float, re.findall(r"-?\d+\.?\d*", block)))
        arrays.append(np.array(numbers))

    # Pair consecutive arrays
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

    return np.array(x_sorted), np.array(y_best)


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
    plt.ylabel("efficiency")
    plt.grid(True)
    plt.legend()

    plt.show()


# Usage
pairs = parse_arrays_from_file("tmp3")
plot_pairs(pairs)
