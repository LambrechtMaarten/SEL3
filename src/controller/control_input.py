"""Data class representing a high-level locomotion command for the brittle star robot."""

import jax.numpy as jnp


class ControlInput:
    """High-level locomotion command consumed by all controller implementations.

    Encapsulates a desired movement direction (angle in world frame) and a
    desired movement speed.  The angle is automatically normalised to the
    range [0, 2π) on construction.

    Attributes:
        speed: Non-negative target speed in simulation units.
        angle: Target heading angle in radians, normalised to [0, 2π).
    """

    def __init__(self, speed: float, angle: float):
        """Initialise a ControlInput instance.

        Args:
            speed: Desired movement speed. Must be >= 0.
            angle: Desired heading in radians. Automatically wrapped to
                [0, 2π).

        Raises:
            ValueError: If ``speed`` is negative.
        """
        if speed < 0:
            raise ValueError("Speed must be >= 0")

        self.speed = speed
        self.angle = angle % (jnp.pi * 2)
