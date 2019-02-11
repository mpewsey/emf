import numpy as np
from ..base import _BasePhase, get_magnetic_perm, get_electric_perm

__all__ = ['Phase2D']


class Phase2D(_BasePhase):
    """
    A class representing a transmission line phase.

    Parameters
    ----------
    name : str
        The name of the phase.
    x, y : float
        The x and y positions of the phase.
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
    def __init__(self, name, x, y, diameter, voltage, current, phase_angle,
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

        self.x = x
        self.y = y

    def magnetic_field(self, x, y, mu0=get_magnetic_perm('air')):
        """
        Calculates the x-y magnetic field vector developed by the phase at
        the specified point. The result is a 2D vector of complex values.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which the magnetic field will
            be calculated.
        mu0 : float
            The magnetic field permeability of the space between
            the phase and point.
        """
        dx = x - self.x
        dy = y - self.y
        r = (dx**2 + dy**2)**0.5

        i = self.phaser_current()
        rw = 0.5 * self.diameter

        if r < rw:
            # Point is inside wire
            b = i * (mu0 / (2*np.pi*rw**2))
        else:
            # Point is outside wire
            b = i * (mu0 / (2*np.pi*r**2))

        return b * np.array([-dy, dx])

    def potential_coeff(self, phase, e0=get_electric_perm('air')):
        """
        Returns the potential coefficient between the phase
        and another phase.

        Parameters
        ----------
        phase : Phase
            Another phase object.
        e0 : float
            The electric permittivity of the space.
        """
        if phase is self:
            a = np.log(4*self.y / self.equiv_diameter())
        else:
            dx2 = (self.x - phase.x)**2
            dy2 = (self.y - phase.y)**2
            skl = (dx2 + dy2)**0.5
            dy2 = (self.y + phase.y)**2
            sklp = (dx2 + dy2)**0.5
            a = np.log(sklp / skl)

        return a/(2*np.pi*e0)
