import numpy as np


def build_uniform_mesh(K, domain=(-1.0, 1.0)):
    """
    Build a uniform 1D mesh.

    Parameters
    ----------
    K : int
        Number of cells/elements.
    domain : tuple of float
        Computational domain (xmin, xmax).

    Returns
    -------
    mesh : dict
        Dictionary containing domain, number of cells, cell size,
        cell edges, and cell centers.
    """
    xmin, xmax = domain

    edges = np.linspace(xmin, xmax, K + 1)
    h = (xmax - xmin) / K
    centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "domain": (xmin, xmax),
        "K": K,
        "h": h,
        "edges": edges,
        "centers": centers,
    }