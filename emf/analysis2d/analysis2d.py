import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..base import _BaseEMFAnalysis

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
    def net_magnetic_field(self, x, y):
        """
        Calculates the result magnetic field caused by all phases.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which the magnetic field will
            be calculated.
        """
        f = sum(p.magnetic_field(x, y, self.mu0) for p in self.phases)
        return sum(x.real**2 + x.imag**2 for x in f)**0.5

    def potential_coeffs(self):
        """
        Returns the potential coefficient matrix of the phases.
        """
        n = len(self.phases)
        p = np.zeros((n, n), dtype='float')

        for i in range(n):
            for j in range(i, n):
                k, l = self.phases[i], self.phases[j]
                p[i, j] = p[j, i] = k.potential_coeff(l, self.e0)

        return p

    def charges(self):
        """
        Returns the charges of the phases.
        """
        p = self.potential_coeffs()
        v = np.array([k.ph_to_gnd_voltage() for k in self.phases])

        return np.dot(np.linalg.inv(p), v)

    def electric_field(self, x, y):
        """
        Returns the electric field vector at the given point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the point where the electric
            field will be calculated.
        """
        qs = self.charges()
        e = np.zeros(2, dtype='complex')

        for p, q in zip(self.phases, qs):
            xm = x - p.x
            a = xm**2 + (p.y - y)**2
            b = xm**2 + (p.y + y)**2
            ex = xm / a - xm / b
            ey = (y - p.y) / a - (y + p.y) / b
            e += (q / (2*np.pi*self.e0)) * np.array([ex, ey])

        return e

    def net_electric_field(self, x, y):
        """
        Returns the resultant electric field at the given point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the point where the electric
            field will be calculated.
        """
        e = self.electric_field(x, y)
        return (e[0].real**2 + e[0].imag**2 + e[1].real**2 + e[1].imag**2)**0.5

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
        v = 0

        for p, q in zip(self.phases, qs):
            dx2 = (p.x - x)**2
            sk = (dx2 + (p.y - y)**2)**0.5
            skp = (dx2 + (p.y + y)**2)**0.5
            v += q * np.log(sk / skp)

        return v / (2*np.pi*self.e0)

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
        return (v.real**2 + v.imag**2)**0.5

    def plot_geometry(self):
        """
        Plots the geometry of the analysis.

        .. figure:: ../_static/geometry.png
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

    def plot_elec_field_profiles(self, xs, ys):
        """
        Plots electric field profiles.

        .. figure:: ../_static/elec_field_profile.png

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Electric Field Profiles',
            xlim=(xs[0], xs[-1]),
            xlabel='X (m)',
            ylabel='Electric Field (V/m)'
        )

        for y in ys:
            f = [self.net_electric_field(x, y) for x in xs]
            label = 'y={} m'.format(y)
            ax.plot(xs, f, label=label)

        ax.set_yscale('log')
        ax.legend()
        ax.grid()

        return ax

    def plot_elec_field_contours(self, xs, ys, cmap='jet'):
        """
        Plots electric field contours.

        .. figure:: ../_static/elec_field_contour.png

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        cmap : str
            The name of the color map to use.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Electric Field (V/m)',
            xlabel='X (m)',
            ylabel='Y (m)',
            aspect='equal'
        )

        r = []
        for x, y in itertools.product(xs, ys):
            f = self.net_electric_field(x, y)
            r.append([x, y, f])

        r = np.array(r)
        mn, mx = np.min(r[:,2]), np.max(r[:,2])
        levels = np.logspace(np.log10(mn), np.log10(mx), 20)
        labels = ['{:.0f}'.format(l) for l in levels[::2]]

        contour = ax.tricontourf(r[:,0], r[:,1], r[:,2],
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(mn, mx)
        )

        cbar = fig.colorbar(contour)
        cbar.ax.set_yticklabels(labels)

        return ax

    def plot_space_potential_profiles(self, xs, ys):
        """
        Plots space potential profiles.

        .. figure:: ../_static/space_potential_profile.png

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
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

    def plot_space_potential_contours(self, xs, ys, cmap='jet'):
        """
        Plots space potential contours.

        .. figure:: ../_static/space_potential_contour.png

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        cmap : str
            The name of the color map to use.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Space Potential (V)',
            xlabel='X (m)',
            ylabel='Y (m)',
            aspect='equal'
        )

        r = []
        for x, y in itertools.product(xs, ys):
            f = self.net_space_potential(x, y)
            r.append([x, y, f])

        r = np.array(r)
        mn, mx = np.min(r[:,2]), np.max(r[:,2])
        levels = np.linspace(mn, mx, 20)

        contour = ax.tricontourf(r[:,0], r[:,1], r[:,2],
            levels=levels,
            cmap=cmap
        )

        fig.colorbar(contour)

        return ax

    def plot_mag_field_profiles(self, xs, ys):
        """
        Plots magnetic field profiles.

        .. figure:: ../_static/mag_field_profile.png

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Magnetic Field Profiles',
            xlim=(xs[0], xs[-1]),
            xlabel='X (m)',
            ylabel='Magnetic Field (T)'
        )

        for y in ys:
            f = [self.net_magnetic_field(x, y) for x in xs]
            label='y={} m'.format(y)
            ax.plot(xs, f, label=label)

        ax.set_yscale('log')
        ax.legend()
        ax.grid()

        return ax

    def plot_mag_field_contours(self, xs, ys, cmap='jet'):
        """
        Plots magnetic field contours.

        .. figure:: ../_static/mag_field_contour.png

        Parameters
        ----------
        xs : array
            An array of x values to plot.
        ys : array
            An array of y values to plot.
        cmap : str
            The name of the color map to use.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Magnetic Field (T)',
            xlabel='X (m)',
            ylabel='Y (m)',
            aspect='equal'
        )

        r = []
        for x, y in itertools.product(xs, ys):
            f = self.net_magnetic_field(x, y)
            r.append([x, y, f])

        r = np.array(r)
        mn, mx = np.min(r[:,2]), np.max(r[:,2])
        levels = np.logspace(np.log10(mn), np.log10(mx), 20)
        labels = ['{:.3g}'.format(l) for l in levels[::2]]

        contour = ax.tricontourf(r[:,0], r[:,1], r[:,2],
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(mn, mx)
        )

        cbar = fig.colorbar(contour)
        cbar.ax.set_yticklabels(labels)

        return ax
