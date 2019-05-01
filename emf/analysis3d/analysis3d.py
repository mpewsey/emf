from __future__ import division
import numpy as np
from math import pi
from ..base import _BaseEMFAnalysis
import matplotlib.pyplot as plt
import matplotlib.colors as colors

__all__ = ['EMFAnalysis3D']


class EMFAnalysis3D(_BaseEMFAnalysis):
    """
    A class for performing electric and magnetic field analysis of
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
    def magnetic_field(self, x):
        """
        Calculates the magnetic field caused by all phases.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinates where the value will be calculated.
        """
        x = np.asarray(x)
        return sum(p.magnetic_field(x, self.mu0) for p in self.phases)

    def net_magnetic_field(self, x):
        """
        Calculates the result magnetic field caused by all phases.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinates where the value will be calculated.
        """
        f = self.magnetic_field(x)
        return np.linalg.norm(f)

    def potential_coeffs(self):
        """
        Returns the potential coefficient matrices of the phases.
        """
        n = len(self.phases)
        fa = np.zeros((n, n), dtype='float')
        fb = np.zeros((n, n), dtype='float')
        sa = np.zeros((n, n), dtype='float')
        sb = np.zeros((n, n), dtype='float')

        for i in range(n):
            for j in range(i, n):
                k, l = self.phases[i], self.phases[j]
                c = k.potential_coeff(l, self.e0)

                fa[i, j] = fa[j, i] = c[0, 0]
                fb[i, j] = fb[j, i] = c[0, 1]
                sa[i, j] = sa[j, i] = c[1, 0]
                sb[i, j] = sb[j, i] = c[1, 1]

        return fa, fb, sa, sb

    def charges(self):
        """
        Returns the charge quantities for the phases.
        """
        fa, fb, sa, sb = self.potential_coeffs()
        v = np.array([k.ph_to_gnd_voltage() for k in self.phases])

        fai = np.linalg.inv(fa)
        dq = np.linalg.inv(sb - sa.dot(fai).dot(fb)).dot(v - sa.dot(fai.dot(v)))
        q = fai.dot(v) - fai.dot(fb.dot(dq))

        return q, dq

    @staticmethod
    def _electric_field(x, b, e, q, dq):
        """
        Calculates the electric field at a point.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinate.
        b, e : array
            The phase segment coordinates.
        q, dq : float
            The charges.
        """
        d1 = np.linalg.norm(e - x)
        d2 = np.linalg.norm(b - x)

        delta = e - b
        l = np.linalg.norm(delta)
        u = delta / l
        proj = np.dot(x - b, u)
        dist = np.linalg.norm(u * proj + b - x)

        c = q - dq * (d1**2 - d2**2)

        ep = c * 2 * l * dist
        ep *= (d1 + d2) / (d1 * d2 * (d1 + d2 - l) * (d1 + d2 + l))
        ep -= dq * (dist / d1 - dist / d2) / l

        epp = c * (1 / d1 - 1 / d2)
        epp -= dq * np.log((d1 + d2 + l) / (d1 + d2 - l)) / l
        epp -= dq * (d1 + d2) * ((d1 - d2)**2 - l**2) / (2 * d1 * d2 * l**2)

        c1 = (b - proj * (e - b) / l - x) / dist
        c2 = (e - b) / l

        return ep * c1 + epp * c2

    def electric_field(self, x):
        """
        Returns the electric field vector at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinates where the value will be calculated.
        """
        x = np.asarray(x)
        e = np.zeros(3, dtype='complex')

        for i, (q, dq) in enumerate(zip(*self.charges())):
            # Phase calculation
            phase = self.phases[i]
            e += self._electric_field(x, phase.x1, phase.x2, q, dq)

            # Reflection calculation
            x1, x2 = phase.images()
            e += self._electric_field(x, x1, x2, -q, -dq)

        return e / (4*pi*self.e0)

    def net_electric_field(self, x):
        """
        Returns the resultant electric field at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinates where the value will be calculated.
        """
        e = self.electric_field(x)
        return np.linalg.norm(e)

    @staticmethod
    def _space_potential(x, b, e, q, dq):
        """
        Calculates the space potential at a point.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinate.
        b, e : array
            The phase segment coordinates.
        q, dq : float
            The charges.
        """
        d1 = np.linalg.norm(e - x)
        d2 = np.linalg.norm(b - x)
        l = np.linalg.norm(e - b)

        c1 = np.log((d1 + d2 + l) / (d1 + d2 - l))
        c2 = (d1 + d2) / l - c1 * (d1**2 + d2**2) / (2*l**2)

        return q * c1 + dq * c2

    def space_potential(self, x):
        """
        Returns the space potential at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinates where the value will be calculated.
        """
        x = np.asarray(x)
        v = 0

        for i, (q, dq) in enumerate(zip(*self.charges())):
            # Phase calculation
            phase = self.phases[i]
            v += self._space_potential(x, phase.x1, phase.x2, q, dq)

            # Reflection calculation
            b, e = phase.images()
            v += self._space_potential(x, b, e, -q, -dq)

        return v / (4*pi*self.e0)

    def net_space_potential(self, x):
        """
        Returns the resultant space potential at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) coordinates where the value will be calculated.
        """
        v = self.space_potential(x)
        return np.linalg.norm(v)

    def plot_geometry(self, symbols={}):
        """
        Plots the geometry of the analysis.

        Parameters
        ----------
        symbols : dict
            A dictionary of plot symbols with any of the following keys:

                * 'lines': The line plot symbol, default is 'b-'
                * 'points': The end point plot symbol, default is 'r.'
                * 'text': The text color, default is 'r'
        """
        # Determine ranges
        x = [p.x1 for p in self.phases]
        x += [p.x2 for p in self.phases]
        x = np.array(x)

        mx = x.max(axis=0)
        c = 0.5 * (mx + x.min(axis=0))
        r = 1.1 * np.max(mx - c)
        xlim, ylim, zlim = np.column_stack([c - r, c + r])

        # Make figure
        fig = plt.figure()

        ax = fig.add_subplot(111,
            projection='3d',
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            xlabel='X',
            ylabel='Y',
            zlabel='Z'
        )

        sym = dict(
            lines='b-',
            points='r.',
            text='r'
        )
        sym.update(symbols)

        for p in self.phases:
            x = np.array([p.x1, p.x2])
            c = np.mean(x, axis=0)

            if sym['lines'] is not None:
                ax.plot(x[:,0], x[:,1], x[:,2], sym['lines'])

            if sym['points'] is not None:
                ax.plot(x[:,0], x[:,1], x[:,2], sym['points'])

            if sym['text'] is not None:
                ax.text(c[0], c[1], c[2], p.name, va='center', ha='center', color=sym['text'])

        return ax

    @staticmethod
    def _plane_points(xs, ys, angle_x, angle_y, angle_z, origin):
        """
        Returns points on the plane surface.

        Parameters
        ----------
        xs, ys : array
            The points along the width and height of the plane.
        angle_x, angle_y, angle_z : float
            The rotation angles about the x, y, and z axes.
        origin : array
            The origin of the view.
        """
        # Create rotation matrix
        if angle_x != 0:
            c, s = np.cos(angle_x), np.sin(angle_x)
            r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        else:
            r = np.identity(3)

        if angle_y != 0:
            c, s = np.cos(angle_y), np.sin(angle_y)
            r = r.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))

        if angle_z != 0:
            c, s = np.cos(angle_z), np.sin(angle_z)
            r = r.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

        # Create mesh and rotate
        p = np.array(np.meshgrid(xs, ys, [0])).T
        p = p.reshape((-1, 3))
        p = p.dot(r.T)
        p += np.asarray(origin)

        return p

    def plot_mag_field_contours(self, xs, ys, angle_x=0, angle_y=0, angle_z=0,
                                origin=(0, 0, 0), cmap='jet'):
        """
        Plots the magnetic field contours.

        Parameters
        ----------
        xs, ys : array
            The points along the width and height of the plane.
        angle_x, angle_y, angle_z : float
            The rotation angles about the x, y, and z axes.
        origin : array
            The origin of the view.
        cmap : str
            The name of the color map to use.
        """
        f = self._plane_points(xs, ys, angle_x, angle_y, angle_z, origin)
        f = np.array([self.net_magnetic_field(p) for p in f]) * 1e7
        x = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Magnetic Field (mG)',
            xlabel="X' (m)",
            ylabel="Y' (m)",
            aspect='equal'
        )

        mn, mx = np.min(f), np.max(f)
        levels = np.logspace(np.log10(mn), np.log10(mx), 20)
        labels = ['{:.3g}'.format(l) for l in levels[::2]]

        contour = ax.tricontourf(x[:,0], x[:,1], f,
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(mn, mx)
        )

        cbar = fig.colorbar(contour)
        cbar.ax.set_yticklabels(labels)

        return ax

    def plot_elec_field_contours(self, xs, ys, angle_x=0, angle_y=0, angle_z=0,
                                 origin=(0, 0, 0), cmap='jet'):
        """
        Plots the magnetic field contours.

        Parameters
        ----------
        xs, ys : array
            The points along the width and height of the plane.
        angle_x, angle_y, angle_z : float
            The rotation angles about the x, y, and z axes.
        origin : array
            The origin of the view.
        cmap : str
            The name of the color map to use.
        """
        f = self._plane_points(xs, ys, angle_x, angle_y, angle_z, origin)
        f = [self.net_electric_field(p) for p in f]
        x = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Electric Field (V/m)',
            xlabel="X' (m)",
            ylabel="Y' (m)",
            aspect='equal'
        )

        mn, mx = np.min(f), np.max(f)
        levels = np.logspace(np.log10(mn), np.log10(mx), 20)
        labels = ['{:.0f}'.format(l) for l in levels[::2]]

        contour = ax.tricontourf(x[:,0], x[:,1], f,
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(mn, mx)
        )

        cbar = fig.colorbar(contour)
        cbar.ax.set_yticklabels(labels)

        return ax

    def plot_space_potential_contours(self, xs, ys, angle_x=0, angle_y=0,
                                      angle_z=0, origin=(0, 0, 0), cmap='jet'):
        """
        Plots the magnetic field contours.

        Parameters
        ----------
        xs, ys : array
            The points along the width and height of the plane.
        angle_x, angle_y, angle_z : float
            The rotation angles about the x, y, and z axes.
        origin : array
            The origin of the view.
        cmap : str
            The name of the color map to use.
        """
        f = self._plane_points(xs, ys, angle_x, angle_y, angle_z, origin)
        f = [self.net_space_potential(p) for p in f]
        x = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

        fig = plt.figure()
        ax = fig.add_subplot(111,
            title='Space Potential (V)',
            xlabel="X' (m)",
            ylabel="Y' (m)",
            aspect='equal'
        )

        mn, mx = np.min(f), np.max(f)
        levels = np.linspace(mn, mx, 20)

        contour = ax.tricontourf(x[:,0], x[:,1], f,
            levels=levels,
            cmap=cmap
        )

        cbar = fig.colorbar(contour)

        return ax
