import jax
from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from src.render import post_render


class Environment:

    def __init__(self, configuration: Configuration):
        simulation_configuration = configuration.simulation
        self.env = BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
            (MJCFBrittleStarMorphology(simulation_configuration.morphology_configuration)),
            (MJCFAquariumArena(simulation_configuration.arena_configuration)),
            simulation_configuration.environment_configuration,
            backend="MJX"
        )

        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)

        self.configuration = configuration
        self.environment_configuration: BrittleStarUndirectedLocomotionEnvironmentConfiguration = simulation_configuration.environment_configuration
        self.arena_configuration = simulation_configuration.arena_configuration
        self.morphology_configuration: BrittleStarMorphologySpecification = simulation_configuration.morphology_configuration

        self.state: BaseEnvState = self._reset(self.configuration.random.split())
        self.action_space = self.env.action_space

    def step(self, action, state=None) -> BaseEnvState:
        if state is None:
            state = self.state
        self.state = self._step(action=action, state=state)
        return self.state

    def reset(self, rng=None):
        if rng is None:
            rng = self.configuration.random.split()
        self.state = self._reset(rng=rng)
        return self.state

    def render(self, state=None):
        if state is None:
            state = self.state
        return post_render(self.env.render(state=state), self.environment_configuration)
