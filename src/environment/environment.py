"""Thin wrapper around the MuJoCo brittle star environment for convenient use."""

import jax
from biorobot.brittle_star.environment.undirected_locomotion.dual import (
    BrittleStarUndirectedLocomotionEnvironment,
)
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from src.render.render import post_render


class Environment:
    """Wrapper around the BrittleStarUndirectedLocomotionEnvironment (MJX backend).

    Builds the MuJoCo environment from the morphology and arena configurations,
    JIT-compiles the step and reset functions, and exposes a simple API for
    stepping, resetting, and rendering.

    Attributes:
        configuration: Global simulation and training configuration.
        env: Underlying MoojoCo environment instance.
        action_space: Action space of the underlying environment.
    """

    def __init__(self, configuration: Configuration):
        """Construct the environment and JIT-compile step and reset.

        Args:
            configuration: Global simulation and training configuration
                containing morphology, arena, and environment settings.
        """
        self.configuration: Configuration = configuration
        self.env = BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
            (MJCFBrittleStarMorphology(self.configuration.simulation.morphology_configuration)),
            (MJCFAquariumArena(self.configuration.simulation.arena_configuration)),
            self.configuration.simulation.environment_configuration,
            backend="MJX",
        )
        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)

        self.action_space = self.env.action_space

    def step(self, action, state) -> BaseEnvState:
        """Advance the simulation by one control timestep.

        Args:
            action: Joint torque array of shape ``(30,)``.
            state: Current environment state.

        Returns:
            Updated :class:`BaseEnvState` after applying the action.
        """
        return self._step(action=action, state=state)

    def reset(self, rng):
        """Reset the environment to an initial state.

        Args:
            rng: JAX random key used to seed the reset.

        Returns:
            Initial :class:`BaseEnvState`.
        """
        return self._reset(rng=rng)

    def render(self, state):
        """Render the current state and return an image array.

        Args:
            state: Environment state to render.

        Returns:
            NumPy image array (BGR) or None if rendering is unavailable.
        """
        return post_render(
            self.env.render(state=state),
            self.configuration.simulation.environment_configuration,
        )
