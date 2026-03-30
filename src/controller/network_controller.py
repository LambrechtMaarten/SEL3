from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from configs.config import Configuration
from configs.subcontrollers.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_state import CPGState
from src.environment.environment import Environment
from src.jax_extra.jax_extra import jarr
from pathlib import Path

NATURAL_DIRECTIONS = [
    0.0,                  # 0°   arm 0 leidt
    2 * jnp.pi / 5,      # 72°  arm 1 leidt
    4 * jnp.pi / 5,      # 144° arm 2 leidt
    6 * jnp.pi / 5,      # 216° arm 3 leidt
    8 * jnp.pi / 5,      # 288° arm 4 leidt
]

TRAINING_DIRECTIONS = [
    (ControlInput.RIGHT, 0.0),
    (ControlInput.UP,    jnp.pi / 2),
    (ControlInput.LEFT,  jnp.pi),
    (ControlInput.DOWN,  3 * jnp.pi / 2),
]

CONTROL_INPUT_TO_ANGLE = {
    ControlInput.RIGHT: 0.0,
    ControlInput.UP:    jnp.pi / 2,
    ControlInput.LEFT:  jnp.pi,
    ControlInput.DOWN:  3 * jnp.pi / 2,
}


class CPGNetwork(nn.Module):
    """Small MLP that maps a direction vector to CPG parameters.

    Attributes:
        num_cpg_params: Number of CPG parameters to output, determined
            by the CPG generator and morphology configuration.
    """

    num_cpg_params: int

    @nn.compact
    def __call__(self, x: jarr) -> jarr:
        """Forward pass through the network.

        Args:
            x: Input vector of shape (2,) containing [cos(θ), sin(θ)].

        Returns:
            JAX array of shape (num_cpg_params,) with CPG parameters.
        """
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_cpg_params)(x)
        return x


def angle_to_vector(theta: float) -> jarr:
    """Converts a direction angle to a unit vector.

    Args:
        theta: Direction angle in radians.

    Returns:
        JAX array of shape (2,) containing [cos(θ), sin(θ)].
    """
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def flatten_params(params) -> jarr:
    """Flattens a nested parameter dict into a single array.

    Args:
        params: Nested parameter dictionary from Flax.

    Returns:
        Flat JAX array containing all network weights.
    """
    leaves, _ = jax.tree_util.tree_flatten(params)
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


def make_unflatten_fn(template):
    """Creates a function that unflattens a flat array into a parameter structure.

    By pre-computing the sizes and treedef outside of vmap, the returned
    function only uses concrete Python integers for slicing, which is
    compatible with jax.vmap and jax.jit.

    Args:
        template: Template parameter structure from Flax init.

    Returns:
        A function that takes a flat JAX array and returns a nested
        parameter dictionary matching the template structure.
    """
    leaves, treedef = jax.tree_util.tree_flatten(template)
    sizes = [int(np.prod(np.array(leaf.shape))) for leaf in leaves]
    shapes = [leaf.shape for leaf in leaves]
    offsets = [int(x) for x in [0] + list(np.cumsum(sizes[:-1]))]

    def unflatten(flat: jarr) -> dict:
        result = [
            flat[offset:offset + size].reshape(shape)
            for offset, size, shape in zip(offsets, sizes, shapes)
        ]
        return jax.tree_util.tree_unflatten(treedef, result)

    return unflatten


class NetworkController(Controller):
    """Controller that uses a neural network to map directions to CPG parameters.

    Instead of storing discrete CPG configurations per direction, this
    controller trains a small MLP that takes a direction vector as input
    and outputs CPG parameters directly. This allows continuous interpolation
    between directions — any angle θ works, not just the trained directions.

    Training happens in two phases:
        1. Pre-training: gradient descent on the 5 natural directions of
           the brittle star using the known optimal CPG parameters rotated
           via symmetry.
        2. Genetic optimization: fine-tuning the network weights to
           interpolate correctly for the 4 cardinal directions.

    Attributes:
        network: The CPGNetwork MLP instance.
        weights: The trained network weights as a flat array, or None
            before training.
        _template_params: Template parameter structure used for
            flattening and unflattening the genome.
    """

    def __init__(self):
        """Initializes the controller with no trained weights."""
        self.network: CPGNetwork | None = None
        self.weights = None
        self._template_params = None

    def _init_network(self, configuration: Configuration):
        """Lazily initializes the network if not already done.

        Args:
            configuration: The current training configuration, used to
                determine the number of CPG parameters.
        """
        if self.network is not None:
            return
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        num_cpg_params = cpg_generator.body_to_jarr(cpg.reset()).size
        self.network = CPGNetwork(num_cpg_params=num_cpg_params)
        self._template_params = self.network.init(
            jax.random.PRNGKey(0), jnp.zeros(2)
        )

    def pretrain_from_cpg(
            self,
            cpg_params_right: jarr,
            configuration: Configuration,
            learning_rate: float = 0.01,
            steps: int = 1000):
        """Pre-trains the network on the 5 natural directions using symmetry.

        Uses the known RIGHT solution and rotates it for each of the 5
        natural directions of the brittle star (every 72°). The network
        learns to map each natural direction to the correct CPG parameters
        before the genetic optimization starts.

        After this pre-training, the genetic optimization will fine-tune
        the network to interpolate between the natural directions for the
        4 cardinal directions.

        Args:
            cpg_params_right: The flat CPG parameter array for the RIGHT
                direction, loaded from a previously trained controller.
            configuration: The current training configuration.
            learning_rate: Learning rate for Adam optimizer.
            steps: Number of gradient descent steps.
        """
        import optax

        self._init_network(configuration)
        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)

        # Bouw training pairs voor de 5 natuurlijke richtingen
        training_pairs = []
        for i, theta in enumerate(NATURAL_DIRECTIONS):
            direction_vector = angle_to_vector(theta)

            # Roteer de CPG-params via symmetrie
            cpg_state = cpg_generator.modulate_body(cpg.reset(), cpg_params_right)
            perm = jnp.roll(jnp.arange(10), shift=i * 2)
            rotated_state = cpg_state.replace(
                amplitude_goals=cpg_state.amplitude_goals[perm],
                offset_goals=cpg_state.offset_goals[perm],
                coupled_phase_biases=cpg_state.coupled_phase_biases[perm][:, perm]
            )
            rotated_params = cpg_generator.body_to_jarr(rotated_state)
            training_pairs.append((direction_vector, rotated_params))
            print(f"Richting {i} ({float(jnp.degrees(theta)):.1f}°): params klaar")

        # Loss = MSE over alle 5 richtingen
        def loss_fn(params):
            total_loss = 0.0
            for direction_vector, target_params in training_pairs:
                output = self.network.apply(params, direction_vector)
                total_loss += jnp.mean((output - target_params) ** 2)
            return total_loss / len(training_pairs)

        # Gradient descent via Adam
        optimizer = optax.adam(learning_rate)
        params = self._template_params
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        print("Start pre-training op 5 natuurlijke richtingen...")
        for i in range(steps):
            params, opt_state, loss = step(params, opt_state)
            if i % 100 == 0:
                print(f"  Stap {i}/{steps}: loss = {float(loss):.6f}")

        self._template_params = params
        self.weights = flatten_params(params)
        print("Pre-training voltooid!")

    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
    ) -> CPGState:
        """Modulates the CPG state using the network output for the given direction.

        Args:
            cpg_state: The current CPG state.
            control_input: The directional input to respond to.
            configuration: The current training configuration.

        Returns:
            A new CPGState modulated with the network's CPG parameters
            for the given direction, or a zeroed modulation for ControlInput.ZZZ.
        """
        self._init_network(configuration)
        cpg_generator = configuration.cpg.cpg_generator

        if control_input == ControlInput.ZZZ:
            return cpg_generator.modulate_body(
                cpg_state,
                cpg_generator.body_to_jarr(
                    cpg_generator.generate(configuration).reset()
                ),
            )

        theta = CONTROL_INPUT_TO_ANGLE.get(control_input, 0.0)
        direction_vector = angle_to_vector(theta)
        unflatten = make_unflatten_fn(self._template_params)
        params = unflatten(self.weights)
        cpg_params = self.network.apply(params, direction_vector)
        return cpg_generator.modulate_body(cpg_state, cpg_params)

    def train_controller(
        self,
        genetic_selections: jarr,
        genetic_evaluations: jarr,
        configuration: Configuration,
    ):
        """Stores the best genome (network weights) from the genetic selections.

        Args:
            genetic_selections: JAX array of selected genomes from the
                final generation.
            genetic_evaluations: JAX array of fitness scores corresponding
                to each selection.
            configuration: The current training configuration.
        """
        self._init_network(configuration)
        best_idx = jnp.argmax(genetic_evaluations)
        self.weights = genetic_selections[best_idx]

    @staticmethod
    def evaluator(configuration: Configuration, rng) -> Callable[[jarr], jarr]:
        """Returns a fitness function that evaluates the network across all directions.

        For each genome (network weights) in the population, simulates
        the brittle star in all training directions and sums the scores.
        Uses jax.vmap to evaluate the full population in parallel.

        Args:
            configuration: The current training configuration.
            rng: A JAX random key for environment reset.

        Returns:
            A callable that maps a population array of shape
            (population_size, genome_size) to a fitness array of
            shape (population_size,).
        """
        cpg_generator = configuration.cpg.cpg_generator
        cpg_template = cpg_generator.generate(configuration)
        num_cpg_params = cpg_generator.body_to_jarr(cpg_template.reset()).size

        dummy_network = CPGNetwork(num_cpg_params=num_cpg_params)
        template_params = dummy_network.init(jax.random.PRNGKey(0), jnp.zeros(2))
        unflatten = make_unflatten_fn(template_params)

        def evaluator(arr: jarr) -> jarr:
            """Evaluates all genomes across all training directions.

            Args:
                arr: Population array of shape (population_size, genome_size).

            Returns:
                Fitness array of shape (population_size,).
            """
            env = Environment(configuration)

            def _evaluate_genome(flat_weights: jarr, _rng: jarr) -> float:
                """Evaluates a single genome across all training directions.

                Args:
                    flat_weights: Flat array of network weights.
                    _rng: A JAX random key for environment reset.

                Returns:
                    Total fitness score summed over all directions.
                """
                params = unflatten(flat_weights)
                total_score = 0.0

                for _, theta in TRAINING_DIRECTIONS:
                    _rng, subkey = jax.random.split(_rng)
                    env_state = env.reset(subkey)

                    cpg = cpg_generator.generate(configuration)
                    direction_vector = angle_to_vector(theta)
                    cpg_params = dummy_network.apply(params, direction_vector)
                    cpg_state = cpg_generator.modulate_body(cpg.reset(), cpg_params)

                    start_x = env_state.observations["disk_position"][0]
                    start_y = env_state.observations["disk_position"][1]
                    expected_dx = jnp.cos(theta)
                    expected_dy = jnp.sin(theta)

                    def step_fn(i, val):
                        """Advances the simulation one step and updates the score.

                        Args:
                            i: Current step index (unused).
                            val: Tuple of (cpg_state, env_state, score).

                        Returns:
                            Updated tuple of (cpg_state, env_state, score).
                        """
                        cpg_state, env_state, score = val
                        cpg_state = cpg.step(cpg_state)
                        env_state = env.step(
                            cpg_generator.outputs_to_actions(
                                cpg_state.outputs, configuration
                            ),
                            env_state,
                        )
                        dx = env_state.observations["disk_position"][0] - start_x
                        dy = env_state.observations["disk_position"][1] - start_y
                        score = score + dx * expected_dx + dy * expected_dy
                        return cpg_state, env_state, score

                    _, _, score = jax.lax.fori_loop(
                        0, 800, step_fn, (cpg_state, env_state, 0.0)
                    )
                    total_score += score

                return total_score

            new_rngs = jax.random.split(rng, len(arr))
            scores = jax.vmap(_evaluate_genome)(arr, new_rngs)
            return scores

        return evaluator

    def genome_size(self, configuration: Configuration) -> int:
        """Returns the total number of network weights.

        Args:
            configuration: The current training configuration.

        Returns:
            Total number of parameters in the CPGNetwork.
        """
        self._init_network(configuration)
        return flatten_params(self._template_params).size

    def save_controller(self, logger: Logger):
        """Saves the trained network weights via the logger.

        Args:
            logger: The logger to write the network weights to.
        """

        flat = np.array(self.weights)
        path = Path(logger.base_folder) / "controller"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Schrijf direct naar bestand, omzeil logger volledig
        with open(path, "w") as f:
            f.write(" ".join([f"{x:.8f}" for x in flat.ravel()]))

    def read_controller(self, path: str):
        """Loads previously trained network weights from a file.

        Args:
            path: Path to the file containing the network weights.
        """
        with open(path, "r") as f:
            values = f.read().split()
            self.weights = jnp.array([float(x) for x in values if x])