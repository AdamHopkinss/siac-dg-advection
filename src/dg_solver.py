import numpy as np
from src.basis import gauss_legendre_quadrature, legendre_vandermonde

def reference_mass_matrix(p, basis="orthonormal_legendre", num_quad=None):
    if basis == "orthonormal_legendre":
        return np.eye(p + 1)

    if num_quad is None:
        num_quad = p + 2

    r_q, w_q = gauss_legendre_quadrature(num_quad)
    V_q = legendre_vandermonde(r_q, p)

    return V_q.T @ (w_q[:, None] * V_q)

def reference_stiffness_matrix(p, num_quad=None):
    """
    Compute S[i, n] = \int_{-1}^1 l_n(r) d l_i/dr(r) dr.

    This is the reference-element volume matrix appearing after
    integration by parts in the weak DG formulation.
    """
    if num_quad is None:
        num_quad = p + 2

    r_q, w_q = gauss_legendre_quadrature(num_quad)

    V_q = legendre_vandermonde(r_q, p, derivative_order=0)
    Vr_q = legendre_vandermonde(r_q, p, derivative_order=1)

    S = Vr_q.T @ (w_q[:, None] * V_q)

    return S

def boundary_basis_values(p):
    """
    Return basis values at r=-1 and r=1.
    """
    V_left = legendre_vandermonde(np.array([-1.0]), p)[0, :]
    V_right = legendre_vandermonde(np.array([1.0]), p)[0, :]

    return V_left, V_right

def lax_friedrichs_flux(u_minus, u_plus, a):
    """
    Lax-Friedrichs flux for linear advection f(u)=a u.

    u_minus : value from the left cell
    u_plus  : value from the right cell
    """
    alpha = abs(a)

    return 0.5 * (a * u_minus + a * u_plus) - 0.5 * alpha * (u_plus - u_minus)

def compute_interface_fluxes_periodic(coeffs, p, a):
    """
    Compute numerical fluxes at all cell-left interfaces with periodic BCs.

    Convention
    ----------
    fluxes[j] is the numerical flux at the left boundary of cell j.

    Therefore:
        fluxes[j]       = flux at interface between cell j-1 and cell j
        fluxes[(j+1)%K] = flux at interface between cell j and cell j+1

    The modulo operations implement periodic wrap-around.
    """
    K = coeffs.shape[0]

    V_left, V_right = boundary_basis_values(p)

    u_left = coeffs @ V_left
    u_right = coeffs @ V_right

    fluxes = np.zeros(K)

    for j in range(K):
        # Interface at the left boundary of cell j:
        # left trace comes from cell j-1, right trace comes from cell j.
        # For j=0, periodicity makes the left neighbor cell K-1.
        left_cell = (j - 1) % K
        right_cell = j

        u_minus = u_right[left_cell]
        u_plus = u_left[right_cell]

        fluxes[j] = lax_friedrichs_flux(u_minus, u_plus, a)

    return fluxes

def dg_rhs_advection(coeffs, mesh, p, a, S=None):
    """
    DG RHS for u_t + a u_x = 0 using weak form and periodic BCs.

    Parameters
    ----------
    coeffs : ndarray, shape (K, p+1)
        Modal DG coefficients.
    mesh : dict
        Mesh dictionary.
    p : int
        Polynomial degree.
    a : float
        Constant advection speed.
    S : ndarray or None
        Optional precomputed reference stiffness matrix.

    Returns
    -------
    rhs : ndarray, shape (K, p+1)
        Time derivative of modal coefficients.
    """
    K = mesh["K"]
    h = mesh["h"]

    if S is None:
        S = reference_stiffness_matrix(p)

    V_left, V_right = boundary_basis_values(p)

    fluxes = compute_interface_fluxes_periodic(coeffs, p, a)

    rhs = np.zeros_like(coeffs)

    for j in range(K):
        flux_left = fluxes[j]
        flux_right = fluxes[(j + 1) % K]

        volume = a * (S @ coeffs[j, :])
        surface = -flux_right * V_right + flux_left * V_left

        rhs[j, :] = (2.0 / h) * (volume + surface)

    return rhs
