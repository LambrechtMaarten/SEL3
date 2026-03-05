import jax
from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

from src.render import post_render

environment_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
    joint_randomization_noise_scale=0.0,
    render_mode="rgb_array",
    simulation_time=10,
    num_physics_steps_per_control_step=10,
    time_scale=2,
    camera_ids=[0, 1],
    render_size=(480, 640)
)
arena_configuration = AquariumArenaConfiguration(
    size=(10, 5), sand_ground_color=False, attach_target=False, wall_height=1.5, wall_thickness=0.1
)
morphology_configuration = default_brittle_star_morphology_specification(
    num_arms=5, num_segments_per_arm=3, use_p_control=True, use_torque_control=False
)

class Environment:

    def __init__(self):
        env_conf, arena_conf, morph_conf = environment_configuration, arena_configuration, morphology_configuration

        self.env = BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
            morphology=(MJCFBrittleStarMorphology(specification=morph_conf)),
            arena=(MJCFAquariumArena(configuration=arena_conf)),
            configuration=env_conf, backend="MJX"
        )
        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)

        self.state = self._reset(rng=jax.random.PRNGKey(seed=0))
        self.action_space = self.env.action_space

        self.environment_configuration = env_conf
        self.arena_configuration = arena_conf
        self.morphology_configuration = morph_conf

    def step(self, action, state=None):
        if state is None:
            state = self.state
        self.state = self._step(action=action, state=state)
        return self.state

    def reset(self, rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(seed=0)
        self.state = self._reset(rng=rng)
        return self.state

    def render(self, state=None):
        if state is None:
            state = self.state
        return post_render(self.env.render(state=state), self.environment_configuration)
