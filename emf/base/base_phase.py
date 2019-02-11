import numpy as np
from .properties import str_property, bool_property

__all__ = ['_BasePhase']


class _BasePhase():
    """
    A base class for creating a transmission line phase.

    Parameters
    ----------
    name : str
        The name of the phase.
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
    name = str_property('name')
    in_deg = bool_property('in_deg')

    TYPES = {
        # Phase type: Phase-to-ground factor
        'ac3': 1/3**0.5,
        'dc': 1,
    }

    def __init__(self, name, diameter, voltage, current, phase_angle,
                 num_wires, spacing, ph_type, in_deg):
        self.name = name
        self.diameter = diameter
        self.voltage = voltage
        self.current = current
        self.phase_angle = phase_angle
        self.num_wires = num_wires
        self.spacing = spacing
        self.ph_type = ph_type
        self.in_deg = in_deg

    def get_phase_angle(self):
        """
        Returns the phase angle in radians.
        """
        if self.in_deg:
            return np.deg2rad(self.phase_angle)
        return self.phase_angle

    def equiv_diameter(self):
        """
        Returns the equivalent diameter for the phase bundle.
        """
        n = self.num_wires
        db = self.spacing / np.sin(np.pi/n)
        return (n * self.diameter * db**(n-1))**(1/n)

    def ph_to_gnd_voltage(self):
        """
        Returns the phase to ground voltage for a 3 phase system.
        The value is a complex number.
        """
        ang = self.get_phase_angle()
        f = _BasePhase.TYPES[self.ph_type]
        return self.voltage * f * complex(np.cos(ang), np.sin(ang))

    def phaser_current(self):
        """
        Returns the currect with real and reactive components.
        The result is a complex number.
        """
        ang = self.get_phase_angle()
        return self.current * complex(np.cos(ang), np.sin(ang))
