import jax

from configs.subconfiguration import SubConfiguration
from src.jax_extra import jarr


class RandomConfiguration(SubConfiguration):
    """
    This class configures the used random seed in the learning algorithm.
    """
    def __init__(self, name, seed: int):
        super().__init__(name)
        self.rng = jax.random.PRNGKey(seed)

    def split(self) -> jarr:
        self.rng, _rng = jax.random.split(self.rng)
        return _rng
