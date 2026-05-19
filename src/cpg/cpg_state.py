import jax
from flax import struct
from flax.typing import Shape
from jax import numpy as jnp

from src.jax_extra.jax_extra import jarr

# T_i = x_i + r_i * cos(F_i) = output
# r_i'' => approaches R_i
# x_i'' => approaches X_i
# F_i' = o_i + sum(j, w_ij * r_j * sin(F_j - F_i - b_ij))


@struct.dataclass
class CPGState:
    time: float
    frequency: float  # (o)
    outputs: jarr  # (T)
    phases: jarr  # (F)
    amplitudes: jarr  # (r)
    d_amplitudes: jarr  # (r')
    amplitude_goals: jarr  # (R)
    offsets: jarr  # (x)
    d_offsets: jarr  # (x')
    offset_goals: jarr  # (X)
    coupled_phase_biases: jarr  # (b)

    @staticmethod
    def reset(num_oscilators: int, phase_biases_shape: Shape):
        return CPGState(
            time=0.0,
            frequency=0.0,
            outputs=jnp.zeros(num_oscilators),
            phases=jnp.zeros(num_oscilators),
            amplitudes=jnp.zeros(num_oscilators),
            offsets=jnp.zeros(num_oscilators),
            d_amplitudes=jnp.zeros(num_oscilators),
            d_offsets=jnp.zeros(num_oscilators),
            amplitude_goals=jnp.zeros(num_oscilators),
            offset_goals=jnp.zeros(num_oscilators),
            coupled_phase_biases=jnp.zeros(phase_biases_shape),
        )

    def modulate_random(self, rng):
        ra, rb, rc = jax.random.split(rng, 3)
        amplitude_goals = jax.random.normal(ra, self.amplitude_goals.shape)
        offset_goals = jax.random.normal(rb, self.offset_goals.shape)
        coupled_phase_biases = jax.random.normal(rc, self.coupled_phase_biases.shape)
        frequency = jnp.pi
        return self.modulate(amplitude_goals, offset_goals, coupled_phase_biases, frequency)

    def modulate(
        self,
        amplitude_goals: jarr,
        offset_goals: jarr,
        coupled_phase_biases: jarr,
        frequency: float,
    ):
        # noinspection PyUnresolvedReferences
        return self.replace(
            amplitude_goals=amplitude_goals,
            offset_goals=offset_goals,
            coupled_phase_biases=coupled_phase_biases,
            frequency=frequency,
        )
