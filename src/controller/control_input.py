import jax.numpy as jnp

class ControlInput:
    def __init__(self, speed: float, angle: float):
        if speed < 0:
            raise ValueError("Speed must be >= 0")

        self.speed = speed
        self.angle = angle % (jnp.pi*2)