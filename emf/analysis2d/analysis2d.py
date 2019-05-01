from __future__ import division
import numpy as np
from math import pi
from ..base import _BaseEMFAnalysis
import matplotlib.pyplot as plt
import matplotlib.colors as colors

__all__ = ['EMFAnalysis2D']


class EMFAnalysis2D(_BaseEMFAnalysis):
    """
    A class for performing electric and magnetic field analysis of
    transmission lines.

    Parameters
    ----------
    phases : list
        A list of :class:`.Phase2D`.
    mu0 : float
        The magnetic permeability of the space.
    e0 : float
        The electric permittivity of the space.
    """
    def magnetic_field(self, x, y):
        """
        Calculates the magnetic field vector caused by all phases.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which the magnetic field will
            be calculated.
        """
        mu0 = self.mu0
        return sum(p.magnetic_field(x, y, mu0) for p in self.phases)

    def net_magnetic_field(self, x, y):
        """
        Calculates the resultant magnetic field caused by all phases.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which the magnetic field will
            be calculated.
        """
        f = self.magnetic_field(x, y)
        return np.linalg.norm(f)

    def potential_coeffs(self):
        """
        Returns the potential coefficient matrix of the phases.
        """
        e0 = self.e0
        phases = self.phases

        n = len(phases)
        p = np.zeros((n, n), dtype='float')

        for i, k in enumerate(phases):
            for j in range(i, n):
                l = phases[j]
                p[i, j] = p[j, i] = k.potential_coeff(l, e0)

        return p

    def charges(self):
        """
        Returns the charges of the phases.
        """
        p = self.potential_coeffs()
        p = np.linalg.inv(p)
        v = [k.ph_to_gnd_voltage() for k in self.phases]

        return p.dot(v)

    def electric_field(self, x, y, qs=None):
        """
        Returns the electric field vector at the given point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the point where the electric
            field will be calculated.
        qs : array
            An array of phase charges. If None, the charges will be calculated.
        """
        if qs is None:
            qs = self.charges()
        else:
            qs = np.asarray(qs)

        qs = qs / (2*pi*self.e0)
        ph = np.array([(p.x, p.y) for p in self.phases])

        xm = x - ph[:,0]
        xm2 = xm**2
        a = xm2 + (ph[:,1] - y)**2
        b = xm2 + (ph[:,1] + y)**2
        ex = xm / a - xm / b
        ey = (y - ph[:,1]) / a - (y + ph[:,1]) / b
        ex = np.dot(qs, ex)
        ey = np.dot(qs, ey)
        e = np.array([ex, ey], dtype='complex')

        return e

    def net_electric_field(self, x, y, qs=None):
        """
        Returns the resultant electric field at the given point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the point where the electric
            field will be calculated.
        qs : array
            An array of phase charges. If None, the charges will be calculated.
        """
        e = self.electric_field(x, y, qs)
        return np.linalg.norm(e)

    def space_potential(self, x, y):
        """
        Returns the space potential at the given point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the point where the space
            potential will be calculated.
        """
        qs = self.charges()
        ph = np.array([(p.x, p.y) for p in self.phases])

        dx2 = (ph[:,0] - x)**2
        sk = dx2 + (ph[:,1] - y)**2
        skp = dx2 + (ph[:,1] + y)**2
        v = np.dot(qs, np.log((sk / skp)**0.5))

        return v / (2*pi*self.e0)

    def net_space_potential(self, x, y):
        """
        Returns the resultant space potential at the given point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the point where the space
            potential will be calculated.
        """
        v = self.space_potential(x, y)
        return np.linalg.norm(v)

    def plot_geometry(self):
        """
        Plots the geometry of the analysis.

        Examples
        --------
        .. plot:: ../examples/analysis2d/geometry.py
            :include-source:
        """
        x = np.array([(p.x, p.y) for p in self.phases])
        xlim = 1.2 * np.array([np.min(x[:,0]), np.max(x[:,0])])
        ylim = 1.2 * np.array([0, np.max(x[:,1])])

        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Phase Geometry',
            xlabel='X (m)',
            ylabel='Y (m)',
            xlim=xlim,
            ylim=ylim,
            aspect='equal'
        )

        ax.grid()
        ax.plot(x[:,0], x[:,1], 'ro')

        for p in self.phases:
            ax.text(p.x, p.y, p.name)

        return ax

    def plot_elec_field_contours(self, xs, ys, cmap='jet'):
        """
        Plots electric field contours.

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        cmap : str
            The name of the color map to use.

        Examples
        --------
        .. plot:: ../examples/analysis2d/elec_field_contours.py
            :include-source:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Electric Field (V/m)',
            xlabel='X (m)',
            ylabel='Y (m)',
            aspect='equal'
        )

        qs = self.charges()
        p = np.array(np.meshgrid(xs, ys)).T
        p = p.reshape(-1, 2)
        f = np.array([self.net_electric_field(x, y, qs) for x, y in p])

        mn, mx = np.min(f), np.max(f)
        levels = np.logspace(np.log10(mn), np.log10(mx), 20)
        labels = ['{:.0f}'.format(l) for l in levels[::2]]

        contour = ax.tricontourf(p[:,0], p[:,1], f,
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(mn, mx)
        )

        cbar = fig.colorbar(contour)
        cbar.ax.set_yticklabels(labels)

        return ax

    def plot_space_potential_contours(self, xs, ys, cmap='jet'):
        """
        Plots space potential contours.

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        cmap : str
            The name of the color map to use.

        Examples
        --------
        .. plot:: ../examples/analysis2d/space_potential_contours.py
            :include-source:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Space Potential (V)',
            xlabel='X (m)',
            ylabel='Y (m)',
            aspect='equal'
        )

        p = np.array(np.meshgrid(xs, ys)).T
        p = p.reshape(-1, 2)
        f = np.array([self.net_space_potential(x, y) for x, y in p])

        mn, mx = np.min(f), np.max(f)
        levels = np.linspace(mn, mx, 20)

        contour = ax.tricontourf(p[:,0], p[:,1], f,
            levels=levels,
            cmap=cmap
        )

        fig.colorbar(contour)

        return ax

    def plot_mag_field_contours(self, xs, ys, cmap='jet'):
        """
        Plots magnetic field contours.

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        cmap : str
            The name of the color map to use.

        Examples
        --------
        .. plot:: ../examples/analysis2d/mag_field_contours.py
            :include-source:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Magnetic Field (mG)',
            xlabel='X (m)',
            ylabel='Y (m)',
            aspect='equal'
        )

        p = np.array(np.meshgrid(xs, ys)).T
        p = p.reshape(-1, 2)
        f = np.array([self.net_magnetic_field(x, y) for x, y in p]) * 1e7

        mn, mx = np.min(f), np.max(f)
        levels = np.logspace(np.log10(mn), np.log10(mx), 20)
        labels = ['{:.3g}'.format(l) for l in levels[::2]]

        contour = ax.tricontourf(p[:,0], p[:,1], f,
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(mn, mx)
        )

        cbar = fig.colorbar(contour)
        cbar.ax.set_yticklabels(labels)

        return ax

    def plot_elec_field_profiles(self, xs, ys):
        """
        Plots electric field profiles.

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.

        Examples
        --------
        .. plot:: ../examples/analysis2d/elec_field_profiles.py
            :include-source:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Electric Field Profiles',
            xlim=(xs[0], xs[-1]),
            xlabel='X (m)',
            ylabel='Electric Field (V/m)'
        )

        qs = self.charges()

        for y in ys:
            f = [self.net_electric_field(x, y, qs) for x in xs]
            label = 'y={} m'.format(y)
            ax.plot(xs, f, label=label)

        ax.set_yscale('log')
        ax.legend()
        ax.grid()

        return ax

    def plot_space_potential_profiles(self, xs, ys):
        """
        Plots space potential profiles.

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.

        Examples
        --------
        .. plot:: ../examples/analysis2d/space_potential_profiles.py
            :include-source:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Space Potential Profiles',
            xlim=(xs[0], xs[-1]),
            xlabel='X (m)',
            ylabel='Space Potential (V)'
        )

        for y in ys:
            f = [self.net_space_potential(x, y) for x in xs]
            label = 'y={} m'.format(y)
            ax.plot(xs, f, label=label)

        ax.legend()
        ax.grid()

        return ax

    def plot_mag_field_profiles(self, xs, ys):
        """
        Plots magnetic field profiles.

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.

        Examples
        --------
        .. plot:: ../examples/analysis2d/mag_field_profiles.py
            :include-source:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Magnetic Field Profiles',
            xlim=(xs[0], xs[-1]),
            xlabel='X (m)',
            ylabel='Magnetic Field (mG)'
        )

        for y in ys:
            f = np.array([self.net_magnetic_field(x, y) for x in xs]) * 1e7
            label='y={} m'.format(y)
            ax.plot(xs, f, label=label)

        ax.set_yscale('log')
        ax.legend()
        ax.grid()

        return ax
