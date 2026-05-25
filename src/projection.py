import numpy as np
from src.basis import legendre_vandermonde, gauss_legendre_quadrature

def l2_project_cell(f, p, x_left, x_right, num_quad=None):
    """
    L2-project f(x) on one physical cell [x_left, x_right]
    using orthonormal Legendre basis on the reference element.

    Returns modal coefficients on the reference basis.
    """
    if num_quad is None:
        num_quad = p + 2

    r_q, w_q = gauss_legendre_quadrature(num_points=num_quad)
    V_q = legendre_vandermonde(eval_nodes=r_q, p=p)

    x_mid = 0.5 * (x_left + x_right)
    h = x_right - x_left

    x_q = x_mid + 0.5 * h * r_q

    f_q = f(x_q)
    # Reference-element projection:
    # c_n = \int_{-1}^1 f(x(r)) l_n(r) dr.
    # Because {l_n} is orthonormal on [-1, 1], the mass matrix is identity
    # and no h/2 factor is needed for these reference modal coefficients.
    coeffs = V_q.T @ (w_q * f_q)

    return coeffs

def l2_project_mesh(f, p, mesh, num_quad=None):
    """
    L2-project f(x) onto a DG space over a 1D mesh.

    Parameters
    ----------
    f : callable
        Function of physical coordinate x.
    p : int
        Polynomial degree.
    mesh : dict
        Mesh dictionary with at least:
            K     - number of elements
            edges - element edges, shape (K+1,)
            h     - mesh size, either scalar or array-like

        Optional:
            domain  - (xmin, xmax)
            centers - element centers

    Returns
    -------
    dg : dict
        DG solution dictionary containing modal coefficients.
    """
    K = mesh["K"]
    edges = mesh["edges"]
    order = p + 1
    coeffs = np.zeros((K, order))
    
    for j in range(K):
        coeffs[j, :] = l2_project_cell(
            f=f, 
            p=p, 
            x_left=edges[j], 
            x_right=edges[j+1], 
            num_quad=num_quad
        )

    dg = {
        "p": p,
        "order": order,
        "K": K,
        "mesh": mesh,
        "coeffs": coeffs,
    }
    
    return dg

#################################### Noise ####################################

def l2_project_cell_noisy(
    f,
    p,
    x_left,
    x_right,
    epsilon,
    rng,
    num_quad=None,
    preserve_cell_average=False,
):
    """
    L2-project noisy initial data f(x) + epsilon * eta
    on one physical cell [x_left, x_right].

    Noise is added at quadrature points before projection.
    """
    if num_quad is None:
        num_quad = p + 2

    r_q, w_q = gauss_legendre_quadrature(num_points=num_quad)
    V_q = legendre_vandermonde(eval_nodes=r_q, p=p)

    x_mid = 0.5 * (x_left + x_right)
    h = x_right - x_left
    x_q = x_mid + 0.5 * h * r_q

    clean_q = f(x_q)
    noise_q = rng.normal(size=clean_q.shape)

    f_q = clean_q + epsilon * noise_q

    coeffs = V_q.T @ (w_q * f_q)

    if preserve_cell_average:
        # coefficient 0 corresponds to the cell-average-like mode
        # if your first orthonormal basis function is constant
        coeffs[0] = V_q[:, 0].T @ (w_q * clean_q)

    return coeffs


def l2_project_mesh_noisy(
    f,
    p,
    mesh,
    q,
    C=1.0,
    seed=None,
    num_quad=None,
    preserve_cell_average=False,
):
    """
    L2-project noisy initial data onto a DG space.

    The noise amplitude is epsilon = C h^q.
    """
    rng = np.random.default_rng(seed)

    K = mesh["K"]
    edges = mesh["edges"]
    order = p + 1
    coeffs = np.zeros((K, order))

    # Works for uniform mesh. If h is array-like, use cellwise h below.
    h_global = mesh["h"]

    for j in range(K):
        h_j = edges[j + 1] - edges[j]
        epsilon_j = C * h_j**q

        coeffs[j, :] = l2_project_cell_noisy(
            f=f,
            p=p,
            x_left=edges[j],
            x_right=edges[j + 1],
            epsilon=epsilon_j,
            rng=rng,
            num_quad=num_quad,
            preserve_cell_average=preserve_cell_average,
        )

    dg = {
        "p": p,
        "order": order,
        "K": K,
        "mesh": mesh,
        "coeffs": coeffs,
        "noise": {
            "type": "physical_before_projection",
            "q": q,
            "C": C,
            "seed": seed,
            "preserve_cell_average": preserve_cell_average,
        },
    }

    return dg


def add_modal_coefficient_noise(dg, sigma, seed=None, preserve_cell_average=True):
    """
    Add Gaussian noise directly to DG modal coefficients.
    """
    rng = np.random.default_rng(seed)

    coeffs = dg["coeffs"]
    noise = sigma * rng.normal(size=coeffs.shape)

    if preserve_cell_average:
        noise[:, 0] = 0.0

    dg_noisy = dg.copy()
    dg_noisy["coeffs"] = coeffs + noise
    dg_noisy["noise"] = {
        "type": "modal",
        "sigma": sigma,
        "seed": seed,
        "preserve_cell_average": preserve_cell_average,
    }

    return dg_noisy