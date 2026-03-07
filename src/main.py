import time

import jax
import jax.numpy as jnp

import \
    configs.logger, \
    configs.simulation_configurations, \
    configs.random_configurations, \
    configs.cpg_configurations, \
    configs.genetic_configurations, \
    configs.controller_configurations

from configs.config import Configuration
from src.cpg.cpg import CPG
from src.env import Environment
from src.jax_extra import jarr


def main():
    configuration = Configuration(
        configs.logger.standard(),
        configs.simulation_configurations.standard(),
        configs.cpg_configurations.standard(),
        configs.random_configurations.standard(),
        configs.genetic_configurations.standard(),
        configs.controller_configurations.standard()
    )

    configuration.logger.log_configuration()

    env = Environment(configuration)
    cpg_generator = configuration.cpg.cpg_generator
    cpg = cpg_generator.generate(configuration)
    # cpg_state = cpg_generator.modulate_cpg(cpg.reset(), 0, env.action_space.high[0] * .5)

    configuration.logger.log_genetic_generation(jnp.zeros(5))
    configuration.logger.log_genetic_generation(jnp.zeros(5))

    genetic_optimizer = configuration.genetic.genetic_optimizer

    def evaluator(arr: jarr, _env: Environment, _cpg: CPG) -> jarr:
        def _evaluator(_arr: jarr) -> jarr:
            _env_state = _env.reset()
            _cpg_state = _cpg.reset().from_jarr(_arr)
            score = 0
            for i in range(200):
                _cpg_state = _cpg.step(_cpg_state)
                _env_state = _env.step(cpg_generator.outputs_to_actions(_cpg_state.outputs, configuration), _env_state)
                score += _env_state.info["xy_distance_from_origin"]
            return score

        return jax.vmap(_evaluator)(arr)

    starting_population = genetic_optimizer.initialize_population(
        configuration.genetic.population_size,
        cpg.reset().to_jarr().size,
        configuration.random.split()
    )
    selection = genetic_optimizer.generation(
        evaluator,
        starting_population,
        configuration.genetic.generations,
        configuration.random.split(),
        env,
        cpg
    )
    cpg_state = cpg.reset().from_jarr(selection[0])
    print(selection[0])

    frames = []
    env_state = env.reset()
    while not (env_state.terminated | env_state.truncated):
        cpg_state = cpg.step(cpg_state)
        actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(actions, env_state)
        frames.append(env.render(env_state))

    configuration.logger.log_video(frames, "video.mp4")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
