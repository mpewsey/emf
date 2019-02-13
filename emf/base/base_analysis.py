from __future__ import division
from .const import get_magnetic_perm, get_electric_perm
from .properties import repr_method

__all__ = ['_BaseEMFAnalysis']


class _BaseEMFAnalysis():
    """
    A base class for performing electric and magnetic field analysis of
    transmission lines.

    Parameters
    ----------
    phases : list
        A list of :class:`.Phase3D`.
    mu0 : float
        The magnetic permeability of the space.
    e0 : float
        The electric permittivity of the space.
    """
    def __init__(self, phases, mu0='air', e0='air'):
        self.phases = phases
        self.set_magnetic_perm(mu0)
        self.set_electric_perm(e0)

    __repr__ = repr_method('phases', 'mu0', 'e0')

    def set_magnetic_perm(self, value):
        """
        Sets the magnetic permeability of the space.

        Parameters
        ----------
        value : str or float
            If a string, the value is looked up from the built-in dictionary.
            Otherwise, the value is assigned to the object.
        """
        if isinstance(value, str):
            value = get_magnetic_perm(value)
        self.mu0 = value

    def set_electric_perm(self, value):
        """
        Sets the electric permittivity of the space.

        Parameters
        ----------
        value : str or float
            If a string, the value is looked up from the built-in dictionary.
            Otherwise, the value is assigned to the object.
        """
        if isinstance(value, str):
            value = get_electric_perm(value)
        self.e0 = value
