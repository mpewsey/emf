import pytest
import numpy as np
from .analysis2d import *
from .phase2d import *


def EMFAnalysis2D_1():
    phases = [
        Phase2D('A', -10, 10.6, 0.033, 525000, 1000, 120, 3, 0.45),
        Phase2D('B', 0, 10.6, 0.033, 525000, 1000, 0, 3, 0.45),
        Phase2D('C', 10, 10.6, 0.033, 525000, 1000, -120, 3, 0.45),
    ]

    return EMFAnalysis2D(phases)


def test_repr():
    repr(EMFAnalysis2D_1())


def test_net_magnetic_field():
    emf = EMFAnalysis2D_1()
    xs = [-200, -100, 0, 100, 200, 500, 1000, 2000, 5000]
    a = np.array([emf.net_magnetic_field(x, 1) for x in xs])
    b = np.array([0.86, 3.47, 210.5, 3.47, 0.86, 0.14, 0.035, 0.009, 0.0014]) / 1e7

    assert pytest.approx(a, 0.04) == b


def test_potential_coeffs():
    emf = EMFAnalysis2D_1()
    a = emf.potential_coeffs().ravel()
    b = [[8.91, 1.53, 0.68], [1.53, 8.91, 1.53], [0.68, 1.53, 8.91]]
    b = np.array(a).ravel()

    assert pytest.approx(a, 0.01) == b


def test_charges():
    emf = EMFAnalysis2D_1()
    a = emf.charges()
    b = [complex(-2.25e-6, 3.19e-6), complex(4.18e-6, 0), complex(-2.25e-6, -3.19e-6)]
    b = np.array(b)

    assert pytest.approx(a.real, 0.01) == b.real
    assert pytest.approx(a.imag, 0.01) == b.imag


def test_electric_field():
    emf = EMFAnalysis2D_1()
    a = emf.electric_field(20, 2)
    b = np.array([complex(-381, -939), complex(1750, 4438)])

    assert pytest.approx(a.real, 0.01) == b.real
    assert pytest.approx(a.imag, 0.01) == b.imag


def test_net_electric_field():
    emf = EMFAnalysis2D_1()
    a = emf.net_electric_field(20, 2)

    assert pytest.approx(a, 0.01) == 4877


def test_space_potential():
    emf = EMFAnalysis2D_1()
    a = emf.space_potential(20, 2)

    assert pytest.approx(a.real, 0.01) == 3535
    assert pytest.approx(a.imag, 0.01) == 8991


def test_net_space_potential():
    emf = EMFAnalysis2D_1()
    a = emf.net_space_potential(20, 2)

    assert pytest.approx(a, 0.01) == 9661


def test_plot_geometry():
    emf = EMFAnalysis2D_1()
    emf.plot_geometry()


def test_plot_elec_field_profiles():
    xs = np.linspace(-20, 20, 500)
    ys = (1, 5, 10.6)

    emf = EMFAnalysis2D_1()
    emf.plot_elec_field_profiles(xs, ys)


def test_plot_mag_field_profiles():
    xs = np.linspace(-20, 20, 500)
    ys = (1, 5, 10.6)

    emf = EMFAnalysis2D_1()
    emf.plot_mag_field_profiles(xs, ys)


def test_plot_space_potential_profiles():
    xs = np.linspace(-20, 20, 500)
    ys = (1, 5, 10.6)

    emf = EMFAnalysis2D_1()
    emf.plot_space_potential_profiles(xs, ys)


def test_plot_elec_field_contours():
    xs = np.linspace(-20, 20, 50)
    ys = np.linspace(0, 40, 50)

    emf = EMFAnalysis2D_1()
    emf.plot_elec_field_contours(xs, ys)


def test_plot_mag_field_contours():
    xs = np.linspace(-20, 20, 50)
    ys = np.linspace(0, 40, 50)

    emf = EMFAnalysis2D_1()
    emf.plot_mag_field_contours(xs, ys)


def test_plot_space_potential_contours():
    xs = np.linspace(-20, 20, 50)
    ys = np.linspace(0, 40, 50)

    emf = EMFAnalysis2D_1()
    emf.plot_space_potential_contours(xs, ys)
