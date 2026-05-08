import os

import jax
import jax.numpy as jnp
import numpy as np

from configs.config import Configuration
from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.controller.controller_configurations import (
    ControllerConfiguration,
)
from configs.subcontrollers.cpg.cpg_configurations import CPGConfiguration
from configs.subcontrollers.genetic.genetic_configurations import GeneticConfiguration
from configs.subcontrollers.logger.logger import Logger
from configs.subcontrollers.random.random_configurations import RandomConfiguration
from configs.subcontrollers.register import register
from configs.subcontrollers.simulation.simulation_configurations import (
    SimulationConfiguration,
)
from src.cpg.cpg_generators.fully_connected_cpg_generator import FullyConnectedCPGGenerator
from src.environment.environment import Environment

ARM_ANGLES_DEG = [0, 72, 144, 216, 288]
ARM_ANGLES_RAD = [k * 2 * jnp.pi / 5 for k in range(5)]
DEMO_SPEEDS = [0.35, 1.0]
EXPERT_STEPS = 300   # stappen voor de expert-video
NN_STEPS = 300       # stappen voor de NN-video (na warm-up)
WARMUP_STEPS = 200   # expert-CPG aandrijving vóór NN overneemt


def render_saved_controller(controller_path: str):
    register()
    configuration = Configuration(
        SubConfigurationMap.get_configuration(Logger, "standard"),
        SubConfigurationMap.get_configuration(SimulationConfiguration, "standard"),
        SubConfigurationMap.get_configuration(CPGConfiguration, "standard"),
        SubConfigurationMap.get_configuration(RandomConfiguration, "standard"),
        SubConfigurationMap.get_configuration(GeneticConfiguration, "crossover"),
        SubConfigurationMap.get_configuration(ControllerConfiguration, "network_pretrain"),
    )
    configuration.logger.init_logger()

    controller = configuration.controller.controller
    controller.read_controller(os.path.join(controller_path, "controller"))

    obs_dim = controller.params["params"]["Dense_0"]["kernel"].shape[0]
    action_dim = controller.params["params"]["Dense_2"]["kernel"].shape[1]
    print(f"Controller geladen: obs_dim={obs_dim}, action_dim={action_dim}")
    if obs_dim != 64:
        print(f"  WAARSCHUWING: obs_dim={obs_dim} i.p.v. 64 — controller opnieuw trainen!")
        return

    env = Environment(configuration)

    # Dezelfde CPG als gebruikt tijdens pretraining
    cpg_generator = FullyConnectedCPGGenerator()
    cpg = cpg_generator.generate(configuration)

    # Laad de snelste gait uit het archief als demo-gait
    archive_path = os.path.join(controller_path, "archive")
    if os.path.exists(os.path.join(archive_path, "selections.npy")):
        selections = np.load(os.path.join(archive_path, "selections.npy"))
        x_positions = np.load(os.path.join(archive_path, "x_positions.npy"))
        best_idx = int(np.argmax(np.abs(x_positions)))
        best_genome = jnp.array(selections[best_idx])
        best_speed = float(np.max(np.abs(x_positions)))
        print(f"Archief gevonden: snelste gait idx={best_idx}, x={x_positions[best_idx]:.3f}")
    else:
        print("Geen archief gevonden naast de controller — gebruik --archive pad/naar/archive")
        return

    @jax.jit
    def expert_cpg_step(cpg_state, env_state):
        cpg_state = cpg.step(cpg_state)
        action = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
        env_state = env.step(action, env_state)
        return cpg_state, env_state, action

    @jax.jit
    def nn_step(params, env_state, angle, speed):
        angle_rel = angle - env_state.observations["disk_rotation"][2]
        obs = controller.build_obs_angle(env_state, angle_rel, speed)
        dist, _ = controller.model.apply(params, obs)
        return dist.mode()

    rng = configuration.random.rng

    # =========================================================
    # VIDEO 1: Expert CPG — wat de robot ZOU moeten doen
    # =========================================================
    print("\n=== Video 1: Expert CPG ===")
    expert_frames = []
    for arm_idx, angle in enumerate(ARM_ANGLES_RAD):
        rng, subkey = jax.random.split(rng)
        env_state = env.reset(subkey)

        expert_cpg_state = cpg_generator.modulate_body(cpg.reset(), best_genome)
        expert_cpg_state = cpg_generator.modulate_symmetric_rotation(
            expert_cpg_state, arm_idx
        )
        for _ in range(EXPERT_STEPS):
            expert_cpg_state, env_state, _ = expert_cpg_step(expert_cpg_state, env_state)
            expert_frames.append(env.render(env_state))

        print(f"  Arm {arm_idx} ({ARM_ANGLES_DEG[arm_idx]}°) klaar")

    configuration.logger.log_video(expert_frames, "expert_cpg.mp4")
    print(f"Expert video opgeslagen ({len(expert_frames)} frames)")

    # =========================================================
    # VIDEO 2: NN na expert warm-up — test of NN de beweging overneemt
    #
    # Redenering: start het NN vanuit een toestand die WEL in de
    # trainingsdistributie zit (robot al in oscillatie). Als het NN
    # de beweging continueert, werkt de pretraining. Als het stopt,
    # is er covariate shift en is PPO nodig.
    # =========================================================
    print("\n=== Video 2: Expert warm-up → NN ===")
    nn_frames = []
    mse_per_segment = []

    for speed in DEMO_SPEEDS:
        for arm_idx, angle in enumerate(ARM_ANGLES_RAD):
            rng, subkey = jax.random.split(rng)
            env_state = env.reset(subkey)
            norm_speed = speed

            # Warm-up: laat expert-CPG de robot in de trainingsdistributie brengen
            expert_cpg_state = cpg_generator.modulate_body(cpg.reset(), best_genome)
            expert_cpg_state = cpg_generator.modulate_symmetric_rotation(
                expert_cpg_state, arm_idx
            )
            for _ in range(WARMUP_STEPS):
                expert_cpg_state, env_state, _ = expert_cpg_step(expert_cpg_state, env_state)
                # Warm-up frames worden NIET opgenomen — we zien alleen het NN

            # NN neemt over vanuit de expert-toestand
            step_errors = []
            for step in range(NN_STEPS):
                # Expert-actie op dit moment (referentie)
                _, _, expert_action = expert_cpg_step(expert_cpg_state, env_state)

                # NN-actie
                nn_action = nn_step(controller.params, env_state, angle, norm_speed)

                # MSE tussen NN en expert op dit moment
                step_errors.append(float(jnp.mean((nn_action - expert_action) ** 2)))

                # Stap de simulatie verder met de NN-actie
                env_state = env.step(nn_action, env_state)
                nn_frames.append(env.render(env_state))

                # Expert ook vooruit stappen voor de volgende referentie
                expert_cpg_state, _, _ = expert_cpg_step(expert_cpg_state, env_state)

            avg_mse = float(np.mean(step_errors))
            mse_per_segment.append(avg_mse)
            print(f"  hoek={ARM_ANGLES_DEG[arm_idx]:3d}°  speed={speed:.2f}  "
                  f"gem. MSE tov expert: {avg_mse:.5f}")

    configuration.logger.log_video(nn_frames, "nn_after_warmup.mp4")
    print(f"\nNN video opgeslagen ({len(nn_frames)} frames)")
    print(f"Gemiddelde MSE over alle segmenten: {float(np.mean(mse_per_segment)):.5f}")
    print(
        "\nInterpretatie MSE:"
        "\n  < 0.01  → uitstekend: NN volgt expert bijna perfect"
        "\n  0.01–0.1 → goed: NN wijkt licht af maar beweegt correct"
        "\n  > 0.1   → slecht: NN wijkt sterk af, PPO nodig"
    )
