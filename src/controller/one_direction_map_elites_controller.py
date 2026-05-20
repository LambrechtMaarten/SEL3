"""CPG-based map-elites controller that builds a diverse archive of locomotion gaits."""

from typing import Any, Callable, Tuple

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


class OneDirectionMapElitesController(Controller):
    """CPG-based controller designed for map-elites quality-diversity optimisation.

    The evaluator scores genomes by their energy efficiency and assigns each
    genome to a behavioural bin based on its x-displacement.  Genomes in the
    same bin compete so that the archive retains a diverse set of gaits across
    different displacement magnitudes.

    Attributes:
        body_cpg: Flat JAX array encoding the CPG body parameters, or None
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
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], Tuple[jarr, jarr]]:
        """Return a map-elites fitness evaluator.

        The evaluator returns both rescaled fitness scores (for map-elites
        selection) and raw energy scores (for logging and archive saving).
        Genomes are grouped into behavioural bins by their x-displacement and
        intra-bin competition is applied so that each bin retains only the
        most energy-efficient gait.

        Args:
            configuration: Global simulation and training configuration.
            rng: JAX random key used to seed environment resets.

        Returns:
            A callable that maps a batch of genome arrays to a tuple of
            (rescaled_scores, raw_energy_scores).
        """

        def evaluator(arr: jarr) -> Tuple[jarr, jarr]:
            env = Environment(configuration)

            def _evaluator(_arr: jarr, _rng: jarr) -> tuple[Any, Any, Any]:
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)

                max_steps = 800

                def step_fn(_, val):
                    cpg_state, env_state, energy = val
                    cpg_state = cpg.step(cpg_state)
                    actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
                    env_state = env.step(
                        actions,
                        env_state,
                    )
                    max_force = env.action_space.high[0]
                    energy += jnp.sum(jnp.pow(max_force, 2) - jnp.pow(jnp.clip(jnp.abs(env_state.observations["actuator_force"]), a_max=max_force), 2))
                    return cpg_state, env_state, energy

                cpg_state, env_state, energy = jax.lax.fori_loop(
                    0, max_steps, step_fn, (cpg_state, env_state, 0)
                )
                return (
                    energy,
                    env_state.observations["disk_position"][0],
                    env_state.observations["disk_position"][1],
                )

            new_rngs = jax.random.split(rng, len(arr))
            energies, x_poss, y_poss = jax.vmap(_evaluator)(arr, new_rngs)
            scores = energies / jnp.abs(x_poss)

            groups = jnp.floor(x_poss * 5).astype(int)
            group_sizes = jnp.count_nonzero(
                jnp.equal(
                    groups,
                    jnp.repeat(groups, groups.shape[0]).reshape((groups.shape[0], groups.shape[0])),
                ),
                axis=0,
            )
            max_scores_in_group = jnp.max(
                jnp.where(groups[:, None] == groups[None, :], scores, -jnp.inf), axis=1
            )
            rescaled_scores = scores / max_scores_in_group
            rescaled_scores = rescaled_scores * (.99**group_sizes)
            jax.debug.log("{y}", y=groups)
            jax.debug.log("{y}", y=scores)

            return rescaled_scores

        return evaluator

    @staticmethod
    def get_edges(configuration: Configuration, rng) -> Callable[[jarr], Tuple[jarr,jarr,jarr]]:
        """Return a function that evaluates genomes and collects per-step joint positions.

        Used after optimisation to record the joint-position trajectories
        (``edges``) for each genome in the archive, together with its
        behavioural bin index.

        Args:
            configuration: Global simulation and training configuration.
            rng: JAX random key used to seed environment resets.

        Returns:
            A callable that maps a batch of genome arrays to a tuple of
            (group_indices, edge_trajectories) where group_indices are integer
            behavioural bin IDs and edge_trajectories has shape
            ``(N, 800, 30)``.
        """

        def evaluator(arr: jarr) -> Tuple[jarr,jarr,jarr]:
            env = Environment(configuration)

            def _evaluator(_arr: jarr, _rng: jarr) -> tuple[Any, Any, Any, Any]:
                env_state = env.reset(_rng)
                cpg_generator = configuration.cpg.cpg_generator
                cpg = cpg_generator.generate(configuration)
                cpg_state = cpg_generator.modulate_body(cpg.reset(), _arr)

                max_steps = 800

                def step_fn(_, val):
                    i, cpg_state, env_state, energy, edges = val
                    cpg_state = cpg.step(cpg_state)
                    actions = cpg_generator.outputs_to_actions(cpg_state.outputs, configuration)
                    max_force = env.action_space.high[0]
                    energy += jnp.sum(jnp.pow(max_force, 2) - jnp.pow(jnp.clip(jnp.abs(env_state.observations["actuator_force"]), a_max=max_force), 2))
                    env_state = env.step(
                        actions,
                        env_state,
                    )

                    return i+1, cpg_state, env_state, energy, edges.at[i].set(env_state.observations["joint_position"])

                i, cpg_state, env_state, energy, edges = jax.lax.fori_loop(
                    0, max_steps, step_fn, (0, cpg_state, env_state, 0, jnp.broadcast_to(env_state.observations["joint_position"], (800,) + env_state.observations["joint_position"].shape))
                )
                return energy, env_state.observations["disk_position"][0], env_state.observations["disk_position"][1], edges

            new_rngs = jax.random.split(rng, len(arr))
            energies, x_poss, y_poss, edges = jax.vmap(_evaluator)(arr, new_rngs)
            groups = jnp.floor(x_poss * 5).astype(int)

            return groups, edges

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
            logger: Logger instance (standard or wandb).
            name: Unused; present for interface compatibility.
        """
        # Standard logger needs string, but wandb does not :(
        # noinspection PyTypeChecker
        logger.log_controller(jnp.array_str(self.body_cpg))

    def read_controller(self, path: str):
        with open(path, "r") as f:
            arrays = f.read()
            self.body_cpg = jnp.array(
                [float(x) for x in arrays.replace("[", " ").replace("]", " ").split()]
            )
