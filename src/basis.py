import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss


def legendre_vandermonde(eval_nodes, p, derivative_order=0):
    """
    Evaluate the orthonormal Legendre basis or its derivatives on [-1, 1].

    Parameters
    ----------
    eval_nodes : array_like
        Evaluation points r_i in [-1, 1].
    p : int
        Maximum polynomial degree.
    derivative_order : int, optional
        0 gives basis values,
        1 gives first derivatives,
        2 gives second derivatives, etc.

    Returns
    -------
    V : ndarray, shape (num_nodes, p+1)
        V[i, n] = d^k l_n / dr^k evaluated at r_i,
        where k = derivative_order.
    """
    r = np.asarray(eval_nodes)
    V = np.zeros((r.size, p + 1))

    for n in range(p + 1):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0

        Pn = Legendre(coeffs)

        if derivative_order > 0:
            Pn = Pn.deriv(derivative_order)

        V[:, n] = np.sqrt((2*n + 1) / 2.0) * Pn(r)

    return V

def gauss_legendre_quadrature(num_points):
    """
    Gauss-Legendre quadrature nodes and weights on [-1, 1].
    """
    return leggauss(num_points)