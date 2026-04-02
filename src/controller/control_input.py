import math
from enum import Enum


class ControlInput(Enum):
    UP = math.pi / 4
    ZZZ = "wait"
    DOWN = math.pi / -2
    LEFT = -math.pi
    RIGHT = 0
    UPLEFT = "upleft"
    UPRIGHT = "upright"
    DOWNLEFT = "downleft"
    DOWNRIGHT = "downright"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
