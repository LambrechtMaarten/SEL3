from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

from configs.subconfiguration_map import SubConfigurationMap
from configs.subcontrollers.simulation.simulation_configurations import SimulationConfiguration
from src.numerical_analysis.euler_solver import EulerSolver


def register():
    SubConfigurationMap.add_configuration(
        SimulationConfiguration,
        SimulationConfiguration(
            "standard",
            EulerSolver(),
            BrittleStarUndirectedLocomotionEnvironmentConfiguration(
                joint_randomization_noise_scale=0.0,
                render_mode="rgb_array",
                simulation_time=20,
                num_physics_steps_per_control_step=10,
                time_scale=2,
                camera_ids=[0, 1],
                render_size=(480, 640),
            ),
            AquariumArenaConfiguration(
                size=(10, 5), sand_ground_color=False, attach_target=False, wall_height=1.5, wall_thickness=0.1
            ),
            default_brittle_star_morphology_specification(
                num_arms=5, num_segments_per_arm=3, use_p_control=True, use_torque_control=False
            ),
        )
    )
