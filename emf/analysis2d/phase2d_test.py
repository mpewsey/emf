from .phase2d import *


def Phase2D_1():
    return Phase2D('A', -10, 10.6, 0.033, 525000, 1000, 120, 3, 0.45)


def test_repr():
    p = Phase2D_1()
    repr(p)
