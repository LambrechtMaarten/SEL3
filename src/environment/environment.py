import jax
from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from src.render.render import post_render


class Environment:

    def __init__(self, configuration: Configuration):
        self.configuration: Configuration = configuration

        self.env = BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
            (MJCFBrittleStarMorphology(self.configuration.simulation.morphology_configuration)),
            (MJCFAquariumArena(self.configuration.simulation.arena_configuration)),
            self.configuration.simulation.environment_configuration,
            backend="MJX"
        )
        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)

        self.action_space = self.env.action_space

    def step(self, action, state) -> BaseEnvState:
        return self._step(action=action, state=state)

    def reset(self, rng):
        return self._reset(rng=rng)

    def render(self, state):
        return post_render(self.env.render(state=state), self.configuration.simulation.environment_configuration)
