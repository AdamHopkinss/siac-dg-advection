import numpy as np

from src.basis import legendre_vandermonde, gauss_legendre_quadrature


def linf_error(u_num, u_exact):
    """
    Discrete L-infinity error on a set of evaluation points.
    """
    return np.max(np.abs(u_num - u_exact))


def relative_linf_error(u_num, u_exact):
    """
    Relative discrete L-infinity error.
    """
    return np.max(np.abs(u_num - u_exact)) / np.max(np.abs(u_exact))


def relative_l2_error_discrete(u_num, u_exact):
    """
    Relative discrete Euclidean error.

    Useful for quick diagnostics on a fixed evaluation grid, but not
    the preferred error for DG convergence studies.
    """
    return np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)


def l2_error_dg(dg, exact_solution, t=0.0, num_quad=None):
    """
    Compute the quadrature-based L2 error of a DG solution.

    Parameters
    ----------
    dg : dict
        DG solution dictionary with entries "coeffs", "p", and "mesh".
    exact_solution : callable
        Function exact_solution(x, t).
    t : float
        Time at which to evaluate the exact solution.
    num_quad : int or None
        Number of Gauss-Legendre quadrature points per cell.

    Returns
    -------
    error : float
        Approximation of ||u_h - u||_{L2}.
    """
    coeffs = dg["coeffs"]
    p = dg["p"]
    mesh = dg["mesh"]

    K = mesh["K"]
    edges = mesh["edges"]

    if num_quad is None:
        num_quad = p + 3

    r_q, w_q = gauss_legendre_quadrature(num_points=num_quad)
    V_q = legendre_vandermonde(eval_nodes=r_q, p=p)

    error_sq = 0.0

    for j in range(K):
        x_left = edges[j]
        x_right = edges[j + 1]

        x_mid = 0.5 * (x_left + x_right)
        h = x_right - x_left

        x_q = x_mid + 0.5 * h * r_q

        u_h_q = V_q @ coeffs[j, :]
        u_exact_q = exact_solution(x_q, t)

        error_sq += 0.5 * h * np.sum(w_q * (u_h_q - u_exact_q)**2)

    return np.sqrt(error_sq)


def relative_l2_error_dg(dg, exact_solution, t=0.0, num_quad=None):
    """
    Compute the relative quadrature-based L2 error.

    Returns
    -------
    rel_error : float
        ||u_h - u||_{L2} / ||u||_{L2}.
    """
    coeffs = dg["coeffs"]
    p = dg["p"]
    mesh = dg["mesh"]

    K = mesh["K"]
    edges = mesh["edges"]

    if num_quad is None:
        num_quad = p + 3

    r_q, w_q = gauss_legendre_quadrature(num_points=num_quad)
    V_q = legendre_vandermonde(eval_nodes=r_q, p=p)

    error_sq = 0.0
    exact_sq = 0.0

    for j in range(K):
        x_left = edges[j]
        x_right = edges[j + 1]

        x_mid = 0.5 * (x_left + x_right)
        h = x_right - x_left

        x_q = x_mid + 0.5 * h * r_q

        u_h_q = V_q @ coeffs[j, :]
        u_exact_q = exact_solution(x_q, t)

        jac = 0.5 * h

        error_sq += jac * np.sum(w_q * (u_h_q - u_exact_q)**2)
        exact_sq += jac * np.sum(w_q * u_exact_q**2)

    return np.sqrt(error_sq / exact_sq)