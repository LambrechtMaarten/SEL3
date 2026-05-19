"""CPG-based controller that drives the brittle star in a single optimised direction."""

from typing import Callable

import jax
from jax import numpy as jnp
from moojoco.environment.base import BaseEnvState

from configs.config import Configuration
from configs.subconfigurations.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr


class OneDirectionController(Controller):
    """CPG-based controller optimised to move the robot as far as possible in one direction.

    A flat genome encodes the full CPG body parameters.  The genetic
    optimizer evolves these parameters to maximise forward displacement along
    the positive x-axis while penalising lateral drift and body rotation.

    Attributes:
        body_cpg: Flat JAX array encoding the CPG body parameters (frequency,
            amplitude goals, offset goals, coupled phase biases), or None
            before a genome has been loaded.
    """

    def __init__(self):
        """Initialise with no CPG genome loaded."""
        self.body_cpg: jarr | None = None

    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
        env_state: BaseEnvState,
    ):
        """Modulate the CPG state using the stored genome and return the updated state.

        Args:
            cpg_state: Current CPG state to be modulated.
            control_input: Unused; present for interface compatibility.
            configuration: Global simulation and training configuration.
            env_state: Unused; present for interface compatibility.

        Returns:
            CPG state modulated with the stored body genome.
        """
        cpg_generator = configuration.cpg.cpg_generator
        return cpg_generator.modulate_body(cpg_state, self.body_cpg)

    @staticmethod
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], jarr]:
        """Return a fitness evaluator that scores genomes by forward displacement.

        The fitness is the cumulative x-displacement over 800 simulation steps
        minus a lateral-drift penalty and a rotation penalty.

        Args:
            configuration: Global simulation and training configuration.
            rng: JAX random key used to seed environment resets.

        Returns:
            A callable that maps a batch of genome arrays to scalar fitness
            scores.
        """

        def evaluator(arr: jarr) -> jarr:
            env = Environment(configuration)

            def _evaluator(_arr: jarr, _rng: jarr) -> jarr | float:
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)

                score = 0.0

                max_steps = 800

                def step_fn(i, val):
                    cpg_state, env_state, score = val
                    cpg_state = cpg.step(cpg_state)
                    env_state = env.step(
                        cpg_generator.outputs_to_actions(cpg_state.outputs, configuration),
                        env_state,
                    )
                    delta_x = env_state.observations["disk_position"][0]
                    # Penalty for movement in wrong direction
                    side_penalty = 0.3 * jnp.abs(env_state.observations["disk_position"][1])
                    rotation_penalty = 0.1 * jnp.abs(env_state.observations["disk_rotation"][0])
                    score = score + delta_x - side_penalty - rotation_penalty

                    return cpg_state, env_state, score

                cpg_state, env_state, score = jax.lax.fori_loop(
                    0, max_steps, step_fn, (cpg_state, env_state, score)
                )
                return score

            new_rngs = jax.random.split(rng, len(arr))
            scores = jax.vmap(_evaluator)(arr, new_rngs)
            return scores

        return evaluator

    def genome_size(self, configuration: Configuration) -> int:
        """Return the number of genes in a single CPG body genome.

        Args:
            configuration: Global simulation and training configuration.

        Returns:
            Integer length of the flat genome array.
        """
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        return cpg_generator.body_to_jarr(cpg.reset()).size

    def save_controller(self, logger: Logger, name: str = "controller"):
        """Log the CPG genome as a string via the provided logger.

        Args:
            logger: Logger instance (standard or wandb).  Note: the standard
                logger expects a string; wandb does not require a special type.
            name: Unused; present for interface compatibility.
        """
        # Standard logger needs string, but wandb does not :(
        # noinspection PyTypeChecker
        logger.log_controller(jnp.array_str(self.body_cpg))

    def read_controller(self, path: str):
        """Load the CPG genome from a plain-text file written by the logger.

        Args:
            path: Filesystem path to the text file containing the genome as
                a space-separated list of floats (with optional square
                brackets).
        """
        with open(path, "r") as f:
            arrays = f.read()
            self.body_cpg = jnp.array(
                [float(x) for x in arrays.replace("[", " ").replace("]", " ").split()]
            )
