import time

import jax.random

from configs.simulation_configs import standard
from src.cpg.cpg_generators import BasicCPGGenerator
from src.env import Environment
from src.render import save_video


def main():
    env = Environment(standard)

    basic_cpg_generator = BasicCPGGenerator(env)
    cpg = basic_cpg_generator.generate()
    cpg.set_state(cpg.state.modulate_random(jax.random.PRNGKey(0)))

    frames = []
    env_state = env.reset()
    while not (env_state.terminated | env_state.truncated):
        actions = basic_cpg_generator.outputs_to_actions(cpg.step().outputs)
        env_state = env.step(actions)
        frames.append(env.render())

    save_video(frames, "../output/video.mp4")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
