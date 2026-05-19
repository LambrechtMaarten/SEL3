# SEL3 - Group 3: Brittle Star Robot

A locomotion control project for a simulated brittle star robot, developed as part of the [Software Engineering Lab 3](https://studiekiezer.ugent.be/studiefiche/nl/C004072/2025) course at Ghent University.

## Overview

The brittle star is a five-armed marine animal capable of flexible, multi-directional movement. This project explores how such a robot can be taught to move in any direction using a combination of:

- **Map-Elites**: an evolutionary algorithm that generates a diverse archive of locomotion gaits using a Central Pattern Generator (CPG)
- **Behavioral Cloning (BC)**: a neural network is pretrained by imitating the best gaits from the archive
- **PPO**: the pretrained network is further fine-tuned using Proximal Policy Optimization, a reinforcement learning algorithm

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt -c constraints.txt
```


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

## Abstract
### From Gaits to Control: A Two-Stage Learning Pipeline for User-Controllable Brittle Star Locomotion
This project proposes a two-stage learning pipeline to achieve user-controllable brittle
star locomotion in MuJoCo simulation, with a primary focus on methodology for
combining evolutionary search and deep reinforcement learning.
In the first stage, a library of discrete gaits is evolved using MAP-Elites, where each gait
encodes a specific locomotion behaviour through Central Pattern Generator (CPG)
parameters. This provides structured, high-quality expert demonstrations that capture
diverse directional and speedmodulated behaviours.
In the second stage, a neural network policy is first pre-trained through behavioral
cloning on these expert demonstrations, then fine-tuned with Proximal Policy
Optimization (PPO). This combination allows the policy to generalise across gaits and
respond smoothly to user-specified direction and speed inputs.
The project demonstrates how evolutionary algorithms and reinforcement learning can
be integrated into a unified training pipeline for controllable locomotion. More broadly, it
explores how evolved behavioural repertoires can serve as effective priors for learning
adaptable control policies in complex robotic systems.

## Poster
<img width="1768" height="2500" alt="brittle_star_poster" src="https://github.ugent.be/user-attachments/assets/157cdadd-84d0-4cc9-94e1-2e03652e129b" />
