# This module configures matplotlib

import os
import sys
import matplotlib

if sys.version_info[0] < 3 and 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['plt']
