from pathlib import Path
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from configs.config import Configuration
from configs.subcontrollers.logger.logger import Logger
from src.controller.control_input import ControlInput
from src.controller.controller import Controller
from src.cpg.cpg_state import CPGState
from src.jax_extra.jax_extra import jarr

# Vijf natuurlijke richtingen (symmetrie van de brittle star)
NATURAL_DIRECTIONS = [
    0.0,                 # 0°   arm 0 leidt
    2 * jnp.pi / 5,      # 72°  arm 1 leidt
    4 * jnp.pi / 5,      # 144° arm 2 leidt
    6 * jnp.pi / 5,      # 216° arm 3 leidt
    8 * jnp.pi / 5,      # 288° arm 4 leidt
]

# Voor manuele besturing via ControlInput (bijv. keyboard / UI)
CONTROL_INPUT_TO_ANGLE = {
    ControlInput.RIGHT: 0.0,
    ControlInput.UP: jnp.pi / 2,
    ControlInput.LEFT: jnp.pi,
    ControlInput.DOWN: 3 * jnp.pi / 2,
}


def angle_to_vector(theta: float) -> jarr:
    """Converts a direction angle to a unit vector [cos θ, sin θ]."""
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def flatten_params(params) -> jarr:
    """Flattens a nested parameter dict into a single array."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


def make_unflatten_fn(template):
    """Creates a function that unflattens a flat array into a parameter structure."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    sizes = [int(np.prod(np.array(leaf.shape))) for leaf in leaves]
    shapes = [leaf.shape for leaf in leaves]
    offsets = [int(x) for x in [0] + list(np.cumsum(sizes[:-1]))]

    def unflatten(flat: jarr) -> dict:
        result = [
            flat[offset : offset + size].reshape(shape)
            for offset, size, shape in zip(offsets, sizes, shapes)
        ]
        return jax.tree_util.tree_unflatten(treedef, result)

    return unflatten


def extract_state_from_cpg(cpg_state: CPGState) -> jarr:
    """Extracts a compact state vector from the CPG state.

    Voor nu gebruiken we de afgevlakte CPG-outputs als interne state.
    Dit kan later uitgebreid worden met environment-observaties.
    """
    return cpg_state.outputs.ravel()


class CPGNetwork(nn.Module):
    """Kleine MLP die (richting, state) naar CPG-parameters mappt.

    In plaats van enkel een richtingsvector te gebruiken, neemt dit netwerk
    ook een state-vector van de brittle star als input. Zo kan het netwerk
    rekening houden met de huidige toestand (bijv. fase, houding) bij het
    kiezen van CPG-parameters.

    Attributes:
        num_cpg_params: Aantal CPG-parameters in de output.
        state_dim: Dimensie van de state-vector.
    """

    num_cpg_params: int
    state_dim: int

    @nn.compact
    def __call__(self, direction_vec: jarr, state_vec: jarr) -> jarr:
        """Forward pass.

        Args:
            direction_vec: JAX array van vorm (2,) met [cos θ, sin θ].
            state_vec: JAX array van vorm (state_dim,) met de huidige state.

        Returns:
            JAX array van vorm (num_cpg_params,) met CPG-parameters.
        """
        x = jnp.concatenate([direction_vec, state_vec], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_cpg_params)(x)
        return x


class NetworkController(Controller):
    """State-afhankelijke controller voor RL: (θ, CPG-state) → CPG-parameters.

    Deze controller bevat:
        - een CPGNetwork dat richting + state naar CPG-parameters mappt;
        - een init-logica om het netwerk op te zetten;
        - optionele pre-training op de 5 natuurlijke richtingen via symmetrie;
        - een act()-functie voor manuele besturing (ControlInput).

    RL (PPO, SAC, …) draait buiten deze klasse en optimaliseert de netwerk-
    parameters op basis van reward (bijv. projectie van verplaatsing op
    gewenste richting).
    """

    def __init__(self):
        self.network: Optional[CPGNetwork] = None
        self._template_params = None
        self._state_dim: Optional[int] = None
        # Voor manuele besturing / runtime gebruik:
        self.weights: Optional[jarr] = None  # flatten_params van params

    def _init_network(self, configuration: Configuration):
        """Initialiseert het netwerk en template-parameters als dat nog niet gebeurd is."""
        if self.network is not None:
            return

        cpg_generator = configuration.cpg.cpg_generator
        cpg = cpg_generator.generate(configuration)
        cpg_state = cpg.reset()

        num_cpg_params = cpg_generator.body_to_jarr(cpg_state).size
        state_dim = extract_state_from_cpg(cpg_state).size
        self._state_dim = int(state_dim)

        self.network = CPGNetwork(
            num_cpg_params=num_cpg_params,
            state_dim=self._state_dim,
        )
        self._template_params = self.network.init(
            jax.random.PRNGKey(0),
            jnp.zeros(2),
            jnp.zeros(self._state_dim),
        )

    # ---------- RL-gerelateerde helpers ----------

    def init_params(self, configuration: Configuration, rng: jarr) -> dict:
        """Geeft een nieuw, random geïnitialiseerd parameter-PyTree terug voor RL.

        RL-algoritmes (PPO, SAC, …) kunnen deze params gebruiken als startpunt
        en ze verder optimaliseren op basis van reward.
        """
        self._init_network(configuration)
        assert self.network is not None
        assert self._state_dim is not None

        params = self.network.init(
            rng,
            jnp.zeros(2),
            jnp.zeros(self._state_dim),
        )
        return params

    def policy_apply(
        self,
        params: dict,
        theta: float,
        cpg_state: CPGState,
    ) -> jarr:
        """Pure policy-call voor RL: (params, θ, CPG-state) → CPG-parameters.

        Dit is de functie die je in je RL-loop gebruikt:
            - direction θ wordt naar [cos θ, sin θ] omgezet;
            - CPG-state wordt naar een state-vector geëxtraheerd;
            - het netwerk produceert CPG-parameters.

        De RL-omgeving kan daarna:
            - cpg_state = modulate_body(cpg_state, cpg_params)
            - cpg_state = cpg.step(cpg_state)
            - env_state = env.step(actions)
        """
        assert self.network is not None
        direction_vec = angle_to_vector(theta)
        state_vec = extract_state_from_cpg(cpg_state)
        return self.network.apply(params, direction_vec, state_vec)

    # ---------- Supervised pre-training (optioneel) ----------

    def pretrain_from_cpg(
        self,
        cpg_params_right: jarr,
        configuration: Configuration,
        learning_rate: float = 0.01,
        steps: int = 1000,
    ):
        """Pre-traint het netwerk op de 5 natuurlijke richtingen via symmetrie.

        We gebruiken de bekende RIGHT-oplossing en roteren die voor elke
        natuurlijke richting (elke 72°). Het netwerk leert om elke natuurlijke
        richting naar de juiste CPG-parameters te mappen, gegeven een neutrale
        state-vector (nulvector). Dit is een eerste stap; RL kan hierna
        fine-tunen op echte rollouts met reward.
        """
        import optax

        self._init_network(configuration)
        assert self._state_dim is not None
        assert self.network is not None

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
                coupled_phase_biases=cpg_state.coupled_phase_biases[perm][:, perm],
            )
            rotated_params = cpg_generator.body_to_jarr(rotated_state)

            # Neutrale state-vector (kan later echte state worden)
            state_vec = jnp.zeros(self._state_dim)

            training_pairs.append((direction_vector, state_vec, rotated_params))
            print(f"Richting {i} ({float(jnp.degrees(theta)):.1f}°): params klaar")

        def loss_fn(params):
            total_loss = 0.0
            for direction_vector, state_vec, target_params in training_pairs:
                output = self.network.apply(params, direction_vector, state_vec)
                total_loss += jnp.mean((output - target_params) ** 2)
            return total_loss / len(training_pairs)

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

    # ---------- Runtime-act voor manuele besturing ----------

    def act(
        self,
        cpg_state: CPGState,
        control_input: ControlInput,
        configuration: Configuration,
    ) -> CPGState:
        """Moduleert de CPG-state met netwerkoutput voor de gegeven richting + state.

        Dit is vooral bedoeld voor manuele besturing (bijv. keyboard):
        - ControlInput.ZZZ → neutrale CPG-configuratie
        - andere inputs → richting naar [cos θ, sin θ], gecombineerd met
          de huidige CPG-state, door het netwerk gestuurd naar CPG-parameters.

        Voor RL gebruik je in plaats hiervan policy_apply() met een θ die
        per episode gesampled wordt.
        """
        self._init_network(configuration)
        assert self._state_dim is not None

        cpg_generator = configuration.cpg.cpg_generator

        if control_input == ControlInput.ZZZ:
            return cpg_generator.modulate_body(
                cpg_state,
                cpg_generator.body_to_jarr(
                    cpg_generator.generate(configuration).reset()
                ),
            )

        if self.weights is None:
            raise RuntimeError(
                "NetworkController.act werd aangeroepen zonder geladen/ingestelde weights. "
                "Gebruik pretrain_from_cpg() of laad RL-getrainde params en zet self.weights."
            )

        theta = CONTROL_INPUT_TO_ANGLE.get(control_input, 0.0)
        direction_vector = angle_to_vector(theta)
        state_vec = extract_state_from_cpg(cpg_state)

        unflatten = make_unflatten_fn(self._template_params)
        params = unflatten(self.weights)
        cpg_params = self.network.apply(params, direction_vector, state_vec)
        return cpg_generator.modulate_body(cpg_state, cpg_params)

    # ---------- Nutsfuncties ----------

    def genome_size(self, configuration: Configuration) -> int:
        """Geeft het totaal aantal netwerkparameters terug.

        Handig als je RL-params als flat vector wil opslaan/loggen.
        """
        self._init_network(configuration)
        return flatten_params(self._template_params).size

    def save_controller(self, logger: Logger):
        """Slaat de huidige netwerkgewichten (self.weights) op via de logger.

        Let op: dit gebruikt de flatten-versie (handig voor logging / snapshots).
        Voor RL kun je ook rechtstreeks het PyTree van params opslaan.
        """
        if self.weights is None:
            raise RuntimeError("Geen weights om op te slaan in NetworkController.")

        flat = np.array(self.weights)
        path = Path(logger.base_folder) / "controller"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(" ".join([f"{x:.8f}" for x in flat.ravel()]))

    def read_controller(self, path: str):
        """Laadt eerder opgeslagen flatten-weights in self.weights.

        Dit is vooral handig voor manuele besturing of om een RL-policy
        te hergebruiken zonder opnieuw te trainen.
        """
        with open(path, "r") as f:
            values = f.read().split()
            self.weights = jnp.array([float(x) for x in values if x])

