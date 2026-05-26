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


####### Functions for projecting sampled functions/data #######

# Since the samples may be noisy and may not align with the DG mesh,
# we use a trapezoidal rule on the sampled data. Mesh edges are inserted
# into the sample grid so that each element integral is computed over the
# full element interval.

def insert_mesh_edges_into_samples(x_data, u_data, edges, tol=1e-14):
    """
    Insert all DG mesh edges into a sampled 1D grid.

    Values at inserted mesh edges are obtained by linear interpolation.
    This avoids extrapolation and ensures that each DG cell integration
    includes both cell endpoints.

    Parameters
    ----------
    x_data : ndarray, shape (N,)
        Original sample locations, assumed strictly increasing.
    u_data : ndarray, shape (N,)
        Sampled values at x_data.
    edges : ndarray, shape (K+1,)
        DG mesh edges to insert.
    tol : float
        Tolerance for avoiding duplicate or nearly duplicate points.

    Returns
    -------
    x_aug : ndarray
        Augmented grid containing the original sample points and all mesh edges.
    u_aug : ndarray
        Sampled/interpolated values on the augmented grid.
    """
    x_data = np.asarray(x_data, dtype=float)
    u_data = np.asarray(u_data, dtype=float)
    edges = np.asarray(edges, dtype=float)

    if x_data.ndim != 1 or u_data.ndim != 1:
        raise ValueError("x_data and u_data must be one-dimensional.")
    if len(x_data) != len(u_data):
        raise ValueError("x_data and u_data must have the same length.")
    if not np.all(np.diff(x_data) > 0):
        raise ValueError("x_data must be strictly increasing.")

    if edges[0] < x_data[0] - tol or edges[-1] > x_data[-1] + tol:
        raise ValueError(
            "Mesh domain must lie inside sampled-data domain. "
            "Otherwise endpoint interpolation would require extrapolation."
        )

    # Combine original points and mesh edges
    x_combined = np.concatenate([x_data, edges])
    x_combined.sort()

    # Remove near-duplicates
    keep = np.ones(len(x_combined), dtype=bool)
    keep[1:] = np.diff(x_combined) > tol
    x_aug = x_combined[keep]

    # Interpolate data values onto augmented grid
    u_aug = np.interp(x_aug, x_data, u_data)

    return x_aug, u_aug

def trapezoidal_weights(x):
    """
    Composite trapezoidal weights for a sorted, possibly nonuniform 1D grid.

    The returned weights satisfy

        np.trapezoid(f, x) == np.sum(trapezoidal_weights(x) * f)

    up to floating-point roundoff.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if len(x) < 2:
        raise ValueError("Need at least two points for trapezoidal quadrature.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    w = np.zeros_like(x, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])

    if len(x) > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])

    return w


def l2_project_sampled_cell_trapezoid(x_cell, u_cell, p, x_left, x_right):
    """
    Project sampled values on one DG cell using trapezoidal quadrature.

    Assumes x_cell includes both x_left and x_right. The physical sample
    points are mapped to the reference element [-1, 1], and the trapezoidal
    rule is applied in reference coordinates.
    """
    x_mid = 0.5 * (x_left + x_right)
    h = x_right - x_left

    r_cell = 2.0 * (x_cell - x_mid) / h
    w_r = trapezoidal_weights(r_cell)

    V = legendre_vandermonde(eval_nodes=r_cell, p=p)

    coeffs = V.T @ (w_r * u_cell)

    return coeffs

def l2_project_sampled_mesh_trapezoid(x_data, u_data, p, mesh):
    """
    L2-project sampled 1D data onto a DG mesh using trapezoidal quadrature.

    Mesh edges are inserted into the sampled grid once as a preprocessing step.
    """
    K = mesh["K"]
    edges = np.asarray(mesh["edges"])
    order = p + 1
    
    # Build augmented sample grid with all DG mesh edges inserted
    x_aug, u_aug = insert_mesh_edges_into_samples(
        x_data=x_data, 
        u_data=u_data, 
        edges=edges
    )
    
    coeffs = np.zeros((K, order))
    
    for j in range(K):
        x_left = edges[j]
        x_right = edges[j+1]
        
        i0 = np.searchsorted(x_aug, x_left, side="left")
        i1 = np.searchsorted(x_aug, x_right, side="right")
        
        x_cell = x_aug[i0:i1]
        u_cell = u_aug[i0:i1]
        
        coeffs[j, :] = l2_project_sampled_cell_trapezoid(
            x_cell=x_cell, 
            u_cell=u_cell, 
            p=p, 
            x_left=x_left, 
            x_right=x_right,
        )
        
    dg = {
        "p": p,
        "order": order,
        "K": K,
        "mesh": mesh,
        "coeffs": coeffs,
        "projection_type": "sampled_trapezoidal_augmented_grid",
        
        # Original sampled data
        "x_sample": np.asarray(x_data).copy(),
        "u_sample": np.asarray(u_data).copy(),
        
        # Augmented data used for projection
        "x_aug": x_aug,
        "u_aug": u_aug,
    }
    
    return dg