from __future__ import division
import numpy as np
from math import pi
from ..base import _BasePhase

__all__ = ['Phase3D']


class Phase3D(_BasePhase):
    """
    A class representing a transmission line phase.

    Parameters
    ----------
    name : str
        The name of the phase.
    x1, x2 : array
        The (x, y, z) coordinates of the beginning and end of the segment.
    diameter : float
        The diameter of the wire.
    voltage : float
        The phase voltage.
    current : float
        The current in the phase.
    phase_angle : float
        The phase angle.
    num_wires : int
        The number of wires in the bundle.
    spacing : float
        The bundle spacing.
    ph_type : {'ac3', 'dc'}
        The phase type. Use 'ac3' for 3-phase alternating current and
        'dc' for direct current.
    in_deg : bool
        Specify True if input phase angle is in degrees; False if angle
        is in radians.
    """
    def __init__(self, name, x1, x2, diameter, voltage, current, phase_angle,
                 num_wires=1, spacing=0, ph_type='ac3', in_deg=True):
        super(Phase3D, self).__init__(
            name=name,
            diameter=diameter,
            voltage=voltage,
            current=current,
            phase_angle=phase_angle,
            num_wires=num_wires,
            spacing=spacing,
            ph_type=ph_type,
            in_deg=in_deg
        )

        self.x1 = np.asarray(x1)
        self.x2 = np.asarray(x2)

    def __repr__(self):
        s = [
            ('name', self.name),
            ('x1', self.x1),
            ('x2', self.x2),
            ('diameter', self.diameter),
            ('voltage', self.voltage),
            ('current', self.current),
            ('phase_angle', self.phase_angle),
            ('num_wires', self.num_wires),
            ('spacing', self.spacing),
            ('ph_type', self.ph_type),
            ('in_deg', self.in_deg)
        ]

        s = ', '.join('{}={!r}'.format(x, y) for x, y in s)
        return '{}({})'.format(type(self).__name__, s)

    @classmethod
    def from_points(cls, name, points, diameter, voltage, current, phase_angle,
                    num_wires=1, spacing=0, ph_type='ac3', in_deg=True):
        """
        Returns a list of phases constructed from a list of points.

        Parameters
        ----------
        name : str
            The name of the phase.
        points : array
            An array of (x, y, z) points for which phases will be constructed.
            The array should be of shape (N, 3).
        diameter : float
            The diameter of the wire.
        voltage : float
            The phase voltage.
        current : float
            The current in the phase.
        phase_angle : float
            The phase angle.
        num_wires : int
            The number of wires in the bundle.
        spacing : float
            The bundle spacing.
        ph_type : {'ac3', 'dc'}
            The phase type. Use 'ac3' for 3-phase alternating current and
            'dc' for direct current.
        in_deg : bool
            Specify True if input phase angle is in degrees; False if angle
            is in radians.
        """
        points = np.asarray(points)
        objs = []

        for x1, x2 in zip(points[:-1], points[1:]):
            obj = cls(
                name=name,
                x1=x1,
                x2=x2,
                diameter=diameter,
                voltage=voltage,
                current=current,
                phase_angle=phase_angle,
                num_wires=num_wires,
                spacing=spacing,
                ph_type=ph_type,
                in_deg=in_deg
            )
            objs.append(obj)

        return objs

    def images(self):
        """
        Returns the ground image end coordinates for the phase.
        """
        r = np.array([1, 1, -1])
        x1 = self.x1 * r
        x2 = self.x2 * r
        return x1, x2

    def length(self):
        """
        Returns the length of the segment.
        """
        return np.linalg.norm(self.x1 - self.x2)

    def magnetic_field(self, x, mu0):
        """
        Returns the magnetic field vector due to the phase.

        Parameters
        ----------
        x : array
            The point at which the field is calculated.
        mu0: float
            The magnetic permeability of the space.
        """
        x = np.asarray(x)
        x1, x2 = self.x1, self.x2
        l = self.length()

        d = np.linalg.norm(x - x1)
        d0 = np.dot(x1 - x, x2 - x1) / l

        c1 = d**2 - d0**2
        c2 = l**2 + 2*l*d0 + d**2

        if d == 0 or c1 == 0 or c2 <= 0:
            return np.zeros(3, dtype='complex')

        i = self.phasor_current()
        k = ((l + d0) / c2**0.5 - d0 / d) / c1
        k *= mu0 / (4*pi*l)

        return i * k * np.cross(x2 - x1, x - x1)

    def _potential_coeff(self, x, phase):
        """
        Returns the potential coefficents between a point and phase.

        Parameters
        ----------
        x : array
            The point for which the potential is calculated.
        phase : :class:`.Phase3D`
            The phase for which the calculation will be performed.
        """
        b, e = phase.x1, phase.x2
        bp, ep = phase.images()

        l = phase.length()
        d1 = np.linalg.norm(e - x)
        d2 = np.linalg.norm(b - x)

        if phase is self:
            r = 0.5 * self.equiv_diameter()
            d1 = (d1**2 + r**2)**0.5
            d2 = (d2**2 + r**2)**0.5

        dp1 = np.linalg.norm(ep - x)
        dp2 = np.linalg.norm(bp - x)

        c1 = np.log((d1 + d2 + l) / (d1 + d2 - l))
        c2 = np.log((dp1 + dp2 - l) / (dp1 + dp2 + l))

        ca = c1 * c2
        cb = ((d1 - d2) - (dp1 - dp2)) / l
        cb += (c1 * (d1**2 - d2**2) - c2 * (dp1**2 - dp2**2)) / (2 * l**2)

        return ca, cb

    def potential_coeff(self, phase, e0):
        """
        Calculates the potential coefficients between the phase potential
        points and another phase segment.
        """
        delta = self.x2 - self.x1
        f = delta / 3 + self.x1
        s = delta * (2/3) + self.x1

        fa, fb = self._potential_coeff(f, phase)
        sa, sb = self._potential_coeff(s, phase)

        return np.array([[fa, fb], [sa, sb]]) / (4*pi*e0)
