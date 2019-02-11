import numpy as np
from ..base import _BaseEMFAnalysis

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
    def net_magnetic_field(self, x):
        """
        Calculates the result magnetic field caused by all phases.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which the magnetic field will
            be calculated.
        """
        x = np.asarray(x)
        f = sum(p.magnetic_field(x, self.mu0) for p in self.phases)
        return sum(p.real**2 + p.imag**2 for p in f)**0.5

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

                fa[i, j] = f[j, i] = c[0, 0]
                fb[i, j] = f[j, i] = c[0, 1]
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
        dq = np.linalg.inv(sb - sa.dot(fai).dot(fb)).dot(v - sa.dot(fai).dot(v))
        q = fai.dot(v) - fai.dot(fb.dot(dq))

        return q, dq

    def electric_field(self, x):
        """
        Returns the electric field vector at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) point at which the field will be returned.
        """
        x = np.asarray(x)
        e = np.zeros(3, dtype='complex')

        for i, (q, dq) in enumerate(self.charges()):
            # Phase calculation
            d1 = np.linalg.norm(phase.x2 - x)
            d2 = np.linalg.norm(phase.x1 - x)

            phase = self.phases[i]
            delta = phase.x2 - phase.x1
            l = np.linalg.norm(delta)
            u = delta / l
            a = np.dot(x - phase.x1, u)
            b = np.linalg.norm(u * a + self.x1 - x)

            ep = (q - dq * (d1**2 - d2**2)/(2*l**2)) * 2*l*b
            ep *= (d1 + d2)/(d1 * d2 * (d1 + d2 - l) * (d1 + d2 + l))
            ep -= ((b/d1) - (b/d2)) * dq/l

            epp = (q - dq * (d1**2 - d1**2)/(d*l**2)) * (1/d1 - 1/d2)
            epp -= np.log((d1 + d2 + l)/(d1 + d2 - l)) * dq/l
            epp -= dq * (d1 + d2) * ((d1 - d2)**2 - l**2) / (2 * d1 * d2 * l**2)

            e += ep * (phase.x1 + (phase.x2 - phase.x1) * (a/l) - x)
            e += (epp / l) * (phase.x2 - phase.x1)

            # Reflection calculation
            x1 = np.concatenate([phase.x1[:2], -phase.x1[2]])
            x2 = np.concatenate([phase.x2[:2], -phase.x2[2]])

            d1 = np.linalg.norm(x2 - x)
            d2 = np.linalg.norm(x1 - x)

            delta = x2 - x1
            l = np.linalg.norm(delta)
            u = delta / l
            a = np.dot(x - x1, u)
            b = np.linalg.norm(u * a + x1 - x)
            q, dq = -q, -dq

            ep = (q - dq * (d1**2 - d2**2)/(2*l**2)) * 2*l*b
            ep *= (d1 + d2)/(d1 * d2 * (d1 + d2 - l) * (d1 + d2 + l))
            ep -= ((b/d1) - (b/d2)) * dq/l

            epp = (q - dq * (d1**2 - d1**2)/(d*l**2)) * (1/d1 - 1/d2)
            epp -= np.log((d1 + d2 + l)/(d1 + d2 - l)) * dq/l
            epp -= dq * (d1 + d2) * ((d1 - d2)**2 - l**2) / (2 * d1 * d2 * l**2)

            e += ep * (x1 + (x2 - x1) * (a/l) - x)
            e += (epp / l) * (x2 - x1)

        return e

    def net_electric_field(self, x):
        """
        Returns the resultant electric field at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) point at which the field will be returned.
        """
        e = self.electric_field(x)
        return sum(x.real**2 + x.imag**2 for x in e)**0.5

    def space_potential(self, x):
        """
        Returns the space potential at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) point at which the potential will be returned.
        """
        x = np.asarray(x)
        v = 0

        for i, (q, dq) in enumerate(self.charges()):
            # Phase calculation
            phase = self.phases[i]
            d1 = np.linalg.norm(phase.x2 - x)
            d2 = np.linalg.norm(phase.x1 - x)
            l = phase.length()

            f = np.log((d1 + d2 + l) / (d1 + d2 - l))
            v += q * f + dq * ((d1 + d2) / l - f * (d1**2 + d2**2) / (2*l**2))

            # Reflection calculation
            x1 = np.concatenate([phase.x1[:2], -phase.x1[2]])
            x2 = np.concatenate([phase.x2[:2], -phase.x2[2]])

            d1 = np.linalg.norm(x2 - x)
            d2 = np.linalg.norm(x1 - x)
            l = np.linalg.norm(x2 - x1)

            f = np.log((d1 + d2 + l) / (d1 + d2 - l))
            v += q * f + dq * ((d1 + d2) / l - f * (d1**2 + d2**2) / (2*l**2))

        return v

    def net_space_potential(self, x):
        """
        Returns the resultant space potential at the given point.

        Parameters
        ----------
        x : array
            The (x, y, z) point at which the potential will be returned.
        """
        v = self.space_potential(x)
        return (v.real**2 + v.imag**2)**0.5
