import numpy as np
from ..base import _BasePhase, array_property, get_magnetic_perm, get_electric_perm

__all__ = ['Phase3D']


def Phase3D(_BasePhase):
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
    x1 = array_property('x1')
    x2 = array_property('x2')

    def __init__(self, name, x1, x2, diameter, voltage, current, phase_angle,
                 num_wires=1, spacing=0, ph_type='ac3', in_deg=True):
        super().__init__(
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

        self.x1 = x1
        self.x2 = x2

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

    def length(self):
        """
        Returns the length of the segment.
        """
        return np.linalg.norm(self.x1 - self.x2)

    def magnetic_field(self, x, mu0=get_magnetic_perm('air')):
        x = np.asarray(x)
        i = self.phaser_current()

        l = np.linalg.norm(x2 - x1)
        d = np.linalg.norm(x - x1)
        d0 = np.dot(x1 - x, x2 - x1) / l
        k = ((l + d0) / (l**2 + 2*l*d0 + d**2)**0.5 - d0 / d) / (d**2 - d0**2)

        return i * (2*k/l) * np.cross(x2 - x1, x - x1)

    def potential_coeff(self, phase, e0=get_electric_perm('air')):
        """
        Calculates the potential coefficients between the phase potential
        points and another phase segment.
        """
        l = phase.length()

        # Calculate potential points
        delta = self.x2 - self.x1
        f = delta / 3 + self.x1
        s = delta * (2/3) + self.x1

        # Segment reflection points
        bp = np.concatenate([phase.x1[:2], -phase.x1[2]])
        ep = np.concatenate([phase.x2[:2], -phase.x2[2]])

        # Calculate point F coefficients
        if phase is self:
            r = 0.5 * self.equiv_diameter()
            d1 = ((l/3)**2 + r**2)**0.5
            d2 = ((2*l/3)**2 + r**2)**0.5
        else:
            d1 = np.linalg.norm(phase.x1 - f)
            d2 = np.linalg.norm(phase.x2 - f)

        dp1 = np.linalg.norm(bp - f)
        dp2 = np.linalg.norm(ep - f)
        fa = np.log((d1+d2+l) / (d1+d2-l)) * np.log((dp1+dp2-l) / (dp1+dp2+l))

        if phase is self:
            fb = 0
        else:
            fb = (d1-d2)/l + np.log((d1+d2+l)/(d1+d2-l)) * (d1**2-d2**2)/(2*l**2)
            fb -= ((dp1-dp2)/l + np.log((dp1+dp2+l)/(dp1+dp2-l)) * (dp1**2-dp2**2)/(2*l**2))

        # Calculate point S coefficients
        if phase is self:
            d1, d2 = d2, d1
        else:
            d1 = np.linalg.norm(phase.x1 - s)
            d2 = np.linalg.norm(phase.x2 - s)

        dp1 = np.linalg.norm(bp - s)
        dp2 = np.linalg.norm(ep - s)
        sa = np.log((d1+d2+l) / (d1+d2-l)) * np.log((dp1+dp2-l) / (dp1+dp2+l))

        if phase is self:
            sb = 0
        else:
            sb = (d1-d2)/l + np.log((d1+d2+l)/(d1+d2-l)) * (d1**2-d2**2)/(2*l**2)
            sb -= ((dp1-dp2)/l + np.log((dp1+dp2+l)/(dp1+dp2-l)) * (dp1**2-dp2**2)/(2*l**2))

        return np.array([[fa, fb], [sa, sb]]) / (4*np.pi*e0)
