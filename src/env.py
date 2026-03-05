from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification


def create_environment() -> BrittleStarUndirectedLocomotionEnvironment:
    morphology = MJCFBrittleStarMorphology(
        specification=morphology_specification
    )
    arena = MJCFAquariumArena(
        configuration=arena_configuration
    )
    env = BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=environment_configuration, backend="MJX"
    )
    return env


environment_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
    joint_randomization_noise_scale=0.0,
    render_mode="rgb_array",
    simulation_time=20,
    num_physics_steps_per_control_step=10,
    time_scale=2,
    camera_ids=[0, 1],
    render_size=(480, 640)
)
arena_configuration = AquariumArenaConfiguration(
    size=(10, 5), sand_ground_color=False, attach_target=False, wall_height=1.5, wall_thickness=0.1
)
morphology_specification = default_brittle_star_morphology_specification(
    num_arms=5, num_segments_per_arm=3, use_p_control=True, use_torque_control=False
)
