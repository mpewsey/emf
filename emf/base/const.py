from math import pi

__all__ = [
    'get_magnetic_perm',
    'get_electric_perm',
]


MAGNETIC_PERM = {
    'air': pi*4e-7,
}

ELECTRIC_PERM = {
    'air': 8.854e-12,
}


def get_magnetic_perm(name):
    """
    Returns the magnetic permeability from the built-in dictionary by name.

    Parameters
    ----------
    name : str
        The name of the space type.
    """
    return MAGNETIC_PERM[name]


def get_electric_perm(name):
    """
    Returns the electric permittivity from the built-in dictionary by name.

    Parameters
    ----------
    name : str
        The name of the space type.
    """
    return ELECTRIC_PERM[name]
