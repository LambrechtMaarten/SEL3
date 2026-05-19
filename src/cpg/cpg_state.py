"""Immutable Flax dataclass holding the full state of a CPG oscillator network."""

import jax
from flax import struct
from flax.typing import Shape
from jax import numpy as jnp

from src.jax_extra.jax_extra import jarr

# CPG oscillator equations (per oscillator i):
# T_i = x_i + r_i * cos(F_i) = output
# r_i'' => approaches R_i
# x_i'' => approaches X_i
# F_i' = o_i + sum(j, w_ij * r_j * sin(F_j - F_i - b_ij))


@struct.dataclass
class CPGState:
    """Immutable snapshot of all CPG oscillator variables at a single timestep.

    Stored as a Flax struct so that the full state can be handled by JAX
    transformations (jit, vmap, scan) without mutation.

    Attributes:
        time: Current simulation time in seconds.
        frequency: Intrinsic oscillation frequency ``o_i`` (shared across all
            oscillators).
        outputs: Oscillator output values ``T_i``, shape ``(N,)``.
        phases: Oscillator phases ``F_i``, shape ``(N,)``.
        amplitudes: Current oscillator amplitudes ``r_i``, shape ``(N,)``.
        d_amplitudes: First derivative of amplitudes ``r_i'``, shape ``(N,)``.
        amplitude_goals: Target amplitude goals ``R_i``, shape ``(N,)``.
        offsets: Current oscillator offsets ``x_i``, shape ``(N,)``.
        d_offsets: First derivative of offsets ``x_i'``, shape ``(N,)``.
        offset_goals: Target offset goals ``X_i``, shape ``(N,)``.
        coupled_phase_biases: Phase bias matrix ``b_ij``, shape ``(N, N)``.
    """

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
        """Create a zeroed CPG state for the given network size.

        Args:
            num_oscilators: Number of oscillators N.
            phase_biases_shape: Shape of the phase bias matrix (typically
                ``(N, N)``).

        Returns:
            A :class:`CPGState` with all arrays initialised to zero and time
            set to 0.0.
        """
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
        """Return a copy of this state with randomly sampled CPG parameters.

        Draws amplitude goals, offset goals, and phase biases independently
        from a standard normal distribution and sets the frequency to π.

        Args:
            rng: JAX random key used to generate the random parameters.

        Returns:
            A new :class:`CPGState` with random CPG parameters applied.
        """
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
        """Return a copy of this state with the given CPG parameters applied.

        Args:
            amplitude_goals: New amplitude goal targets, shape ``(N,)``.
            offset_goals: New offset goal targets, shape ``(N,)``.
            coupled_phase_biases: New phase bias matrix, shape ``(N, N)``.
            frequency: New intrinsic frequency scalar.

        Returns:
            A new :class:`CPGState` with the provided parameters replacing the
            current ones.
        """
        # noinspection PyUnresolvedReferences
        return self.replace(
            amplitude_goals=amplitude_goals,
            offset_goals=offset_goals,
            coupled_phase_biases=coupled_phase_biases,
            frequency=frequency,
        )
