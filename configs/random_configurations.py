import jax

from configs.subconfiguration import SubConfiguration
from src.jax_extra import jarr


class RandomConfiguration(SubConfiguration):
    def __init__(self, name, rng: jarr):
        super().__init__(name)
        self.rng = rng

    def split(self) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return _rng

standard = RandomConfiguration(
    "standard",
    jax.random.PRNGKey(0),
)