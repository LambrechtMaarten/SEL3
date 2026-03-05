import time

import jax.numpy as jnp

from configs.simulation_configs import standard
from src.cpg import create_cpg, map_cpg_outputs_to_actions
from src.env import Environment
from src.render import save_video

environment_configuration, arena_configuration, morphology_configuration = standard


def main():
    env = Environment(environment_configuration, arena_configuration, morphology_configuration)

    cpg = create_cpg(environment_configuration.control_timestep)
    cpg_state = cpg.reset().modulate_random()

    amplitude_goals = jnp.ones(10).at[1::2].set(.2)
    offset_goals = jnp.zeros(10)
    coupled_phase_biases = jnp.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if i == j - 1:
                coupled_phase_biases = coupled_phase_biases.at[i, j].set(-jnp.pi)
            if i == j + 1:
                coupled_phase_biases = coupled_phase_biases.at[i, j].set(jnp.pi + i / 10)
    frequency = jnp.pi

    cpg_state = cpg_state.modulate(amplitude_goals, offset_goals, coupled_phase_biases, frequency)

    frames = []
    env_state = env.reset()
    while not (env_state.terminated | env_state.truncated):
        cpg_state = cpg.step(state=cpg_state)
        actions = map_cpg_outputs_to_actions(morphology_configuration, cpg_state=cpg_state)
        env_state = env.step(actions)
        frame = env.render()
        frames.append(frame)

    save_video(frames, "../output/video.mp4")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
