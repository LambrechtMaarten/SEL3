import time

import jax.random

from configs.simulation_configs import standard
from src.cpg.cpg_generators import BasicCPGGenerator, CPGGenerator
from src.env import Environment
from src.genetic_optimization import BasicGeneticOptimizer
from src.jax_extra import jarr
from src.render import save_video


def main():
    env = Environment(standard)

    cpg_generator = BasicCPGGenerator(env)
    cpg = cpg_generator.generate()
    cpg_state = cpg_generator.modulate_cpg(cpg.reset(), 3, env.action_space.high[0] * .25)
    cpg.set_state(cpg_state)

    def evaluator(arr: jarr):
        env.reset()
        cpg.set_state(cpg_generator.from_jarr(cpg.reset(), arr))
        score = 0
        for i in range(200):
            env_state = env.step(cpg_generator.outputs_to_actions(cpg.step().outputs))
            score += env_state.reward
        return score

    genetic_optimizer = BasicGeneticOptimizer(evaluator, 10, cpg_generator.to_jarr(cpg_state).size, jax.random.PRNGKey(1))

    population = genetic_optimizer.initialize_population()
    for i in range(50):
        evaluations = genetic_optimizer.evaluate_population(population)
        selections = genetic_optimizer.select(population, evaluations)
        jax.debug.print("{i}\t{x}",i=i, x=evaluations[0])
        population = genetic_optimizer.reproduce(selections, evaluations)

    cpg.set_state(cpg_generator.from_jarr(cpg.reset(), population[0]))
    frames = []
    env_state = env.reset()
    while not (env_state.terminated | env_state.truncated):
        actions = cpg_generator.outputs_to_actions(cpg.step().outputs)
        env_state = env.step(actions)
        frames.append(env.render())

    save_video(frames, "../output/video.mp4")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
