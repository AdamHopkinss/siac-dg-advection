import numpy as np

from src.basis import legendre_vandermonde


def make_evaluation_grid(mesh, points_per_cell):
    """
    Create uniform reference and physical evaluation points in each cell.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary with entries "K" and "edges".
    points_per_cell : int
        Number of evaluation points per cell.

    Returns
    -------
    r_eval : ndarray, shape (points_per_cell,)
        Reference evaluation nodes in [-1, 1].
    x_eval : ndarray, shape (K, points_per_cell)
        Physical evaluation nodes in each cell.
    """
    K = mesh["K"]
    edges = mesh["edges"]
    centers = mesh["centers"]

    r_eval = np.linspace(-1.0, 1.0, points_per_cell)
    x_eval = np.zeros((K, points_per_cell))

    for j in range(K):
        x_left = edges[j]
        x_right = edges[j + 1]

        x_mid = centers[j]  # 0.5 * (x_left + x_right)
        h = x_right - x_left
        
        # Reference-to-physical map:
        # x(r) = x_mid + (h/2) r, mapping [-1,1] to [x_left,x_right].
        x_eval[j, :] = x_mid + 0.5 * h * r_eval

    return r_eval, x_eval


def evaluate_dg_solution(dg, points_per_cell):
    """
    Evaluate a modal DG solution at uniform points inside each cell.

    Parameters
    ----------
    dg : dict
        DG solution dictionary with entries "coeffs", "p", and "mesh".
    points_per_cell : int
        Number of evaluation points per cell.

    Returns
    -------
    x_eval_flat : ndarray, shape (K * points_per_cell,)
        Physical evaluation points.
    u_eval_flat : ndarray, shape (K * points_per_cell,)
        DG solution evaluated at those points.
    """
    coeffs = dg["coeffs"]
    p = dg["p"]
    mesh = dg["mesh"]
    K = mesh["K"]

    r_eval, x_eval = make_evaluation_grid(mesh, points_per_cell)
    V_eval = legendre_vandermonde(r_eval, p)

    u_eval = np.zeros((K, points_per_cell))

    for j in range(K):
        u_eval[j, :] = V_eval @ coeffs[j, :]

    return x_eval.ravel(), u_eval.ravel()