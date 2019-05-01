import pytest
import numpy as np
from math import pi
from .analysis3d import *
from .phase3d import *


def EMFAnalysis3D_1():
    phases = [
        Phase3D('A', [0, -150, 10], [0, 150, 10], 0.033, 525000, 1000, 0, 3, 0.45),
    ]

    return EMFAnalysis3D(phases)


def EMFAnalysis3D_2():
    phases = [
        Phase3D('A', [-10, -150, 10], [-10, 150, 10], 0.033, 525000, 1000, 0, 3, 0.45),
        Phase3D('B', [0, -150, 10], [0, 150, 10], 0.033, 525000, 1000, 120, 3, 0.45),
        Phase3D('C', [10, -150, 10], [10, 150, 10], 0.033, 525000, 1000, -120, 3, 0.45),
    ]

    return EMFAnalysis3D(phases)


def test_net_magnetic_field():
    emf = EMFAnalysis3D_1()
    xs = [0]
    a = np.array([emf.net_magnetic_field([x, 0, 1]) for x in xs])
    b = np.array([221]) / 1e7

    assert pytest.approx(a, 0.01) == b


def test_plot_geometry():
    emf = EMFAnalysis3D_1()
    emf.plot_geometry()


def test_plot_mag_field_contours():
    emf = EMFAnalysis3D_2()
    emf.plot_mag_field_contours(
        xs=np.linspace(-20, 20, 25),
        ys=np.linspace(0, 40, 25),
        angle_x=pi/2
    )
