import time

import jax

from configs import simulation_configurations, random_configurations, cpg_configurations
from configs.config import Configuration
from src.env import Environment
from src.genetic_optimization import BasicGeneticOptimizer
from src.jax_extra import jarr
from src.render import save_video


def main():
    configuration = Configuration(
        simulation_configurations.standard,
        cpg_configurations.standard,
        random_configurations.standard,
    )

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate()
    cpg.set_state(cpg_generator.modulate_cpg(cpg.reset(), 0, env.action_space.high[0] * .5))

    def evaluator(arr: jarr):
        env.reset()
        cpg.set_state(cpg.reset().from_jarr(arr))
        score = 0
        for i in range(200):
            env_state = env.step(cpg_generator.outputs_to_actions(cpg.step().outputs))
            score += env_state.info["xy_distance_from_origin"]
        return score

    genetic_optimizer = BasicGeneticOptimizer(evaluator, 2, cpg.reset().to_jarr().size, jax.random.PRNGKey(1))

    selection = genetic_optimizer.generation(genetic_optimizer.initialize_population(), 2)

    cpg.set_state(cpg.reset().from_jarr(selection[0]))
    print(selection[0])
    frames = []
    env_state = env.reset()
    while not (env_state.terminated | env_state.truncated):
        actions = cpg_generator.outputs_to_actions(cpg.step().outputs)
        env_state = env.step(actions)
        frames.append(env.render())
        # jax.debug.log("{x}",x=env_state.info)

    save_video(frames, "../output/video2.mp4")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
