"""
=================================
Base Components (:mod:`emf.base`)
=================================

This module contains base components used by other modules.

Constants
=========
.. autosummary::
    :toctree: generated/

    get_magnetic_perm
    get_electric_perm


Properties
==========
.. autosummary::
    :toctree: generated/

    bool_property
    str_property
    array_property
    repr_method


Base Classes
============
.. autosummary::
    :toctree: generated/

    _BaseEMFAnalysis
    _BasePhase
"""

from .base_analysis import *
from .base_phase import *
from .const import *
from .properties import *
