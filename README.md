# SEL3 - Group 3: Brittle Star Robot

A locomotion control project for a simulated brittle star robot, developed as part of the Software Engineering Lab 3 course at Ghent University.

## Overview

The brittle star is a five-armed marine animal capable of flexible, multi-directional movement. This project explores how such a robot can be taught to move in any direction using a combination of:

- **Map-Elites**: an evolutionary algorithm that generates a diverse archive of locomotion gaits using a Central Pattern Generator (CPG)
- **Behavioral Cloning (BC)**: a neural network is pretrained by imitating the best gaits from the archive
- **PPO**: the pretrained network is further fine-tuned using Proximal Policy Optimization, a reinforcement learning algorithm

## Usage

```bash
# Step 1: generate a gait archive with map-elites
python -m src.main map-elites

# Step 2: pretrain the network on the archive, then fine-tune with PPO
python -m src.main train_network_pretrain <path/to/archive>

# Step 3: render a video of the trained controller
python -m src.main render <path/to/controller>

```

## Documentation

Full API documentation is available at:
**[https://github.ugent.be/pages/jbatens/SEL3-2026-Groep-3/](https://github.ugent.be/pages/jbatens/SEL3-2026-Groep-3/)**
