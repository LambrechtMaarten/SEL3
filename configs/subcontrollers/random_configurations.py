import jax

from configs.subconfiguration import SubConfiguration
from src.jax_extra import jarr


class RandomConfiguration(SubConfiguration):
    def __init__(self, name, seed: int):
        super().__init__(name)
        self.rng = jax.random.PRNGKey(seed)

    def split(self) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return _rng


standard = lambda: RandomConfiguration(
    "standard",
    0,
)
