import jax
import jax.numpy as jnp

from src.environment.environment import Environment
from src.controller.network_controller import NetworkController

def run_rl_training(configuration):
    logger = configuration.logger  # jouw wandb-logger

    controller: NetworkController = configuration.controller.controller
    rng = jax.random.PRNGKey(0)

    # Init policy parameters
    params = controller.init_params(configuration, rng)

    env = Environment(configuration)
    cpg_gen = configuration.cpg.cpg_generator

    NUM_EPISODES = 10          # snel testen
    EPISODE_LENGTH = 200       # aantal stappen per episode

    for episode in range(NUM_EPISODES):
        rng, key_theta, key_env = jax.random.split(rng, 3)
        theta = jax.random.uniform(key_theta, (), minval=0.0, maxval=2*jnp.pi)

        env_state = env.reset(key_env)
        cpg = cpg_gen.generate(configuration)
        cpg_state = cpg.reset()

        start_x = env_state.observations["disk_position"][0]
        start_y = env_state.observations["disk_position"][1]

        episode_reward = 0.0

        for t in range(EPISODE_LENGTH):
            # Policy
            cpg_params = controller.policy_apply(params, theta, cpg_state)

            # CPG update
            cpg_state = cpg_gen.modulate_body(cpg_state, cpg_params)
            cpg_state = cpg.step(cpg_state)

            # Environment update
            actions = cpg_gen.outputs_to_actions(cpg_state.outputs, configuration)
            env_state = env.step(actions, env_state)

            # Reward
            dx = env_state.observations["disk_position"][0] - start_x
            dy = env_state.observations["disk_position"][1] - start_y
            reward = dx * jnp.cos(theta) + dy * jnp.sin(theta)

            episode_reward += float(reward)

            # Log per timestep
            logger.log({"reward": float(reward), "timestep": t})

        # Log per episode
        logger.log({"episode_reward": episode_reward, "episode": episode})

    print("RL-testtraining klaar.")
