import jax
from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

from src.render import post_render

class Environment:

    def __init__(self, env_conf, arena_conf, morph_conf):
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