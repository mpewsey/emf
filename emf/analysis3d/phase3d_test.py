from .phase3d import *


def Phase3D_1():
    return Phase3D('A', [0, -150, 10], [0, 150, 10], 0.033, 525000, 1000, 0, 3, 0.45)


def test_repr():
    p = Phase3D_1()
    repr(p)
