import numpy as np
import math

from scipy.special import binom
import scipy.linalg
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve

from scipy.interpolate import BSpline

from src.basis import legendre_vandermonde, gauss_legendre_quadrature
from src.evaluation import make_evaluation_grid

def siac_cgam(moments: int, BSorder: int):
    """
    Computes the SIAC coefficients for a symmetric kernel by enforcing
    only the even moment conditions.

    Returns
    -------
    cgam : array of length moments+1
        Expanded coefficient vector in the old ordering:
        [c_{-RS}, ..., c_{-1}, c_0, c_1, ..., c_RS]
    """
    assert moments % 2 == 0, "moments should be even!"
    
    RS = int(np.ceil(moments / 2))
    numspline = moments + 1
    R = RS + 1
    
    A = np.zeros((R, R), dtype=float)
    
    even_moments = np.arange(0, moments+1, 2)
    
    for row, m in enumerate(even_moments):
        for gam in range(R):
            
            component = 0.0
            
            if gam == 0:
                shifts = [0]
            else:
                shifts = [-gam, gam]
                
            for shift in shifts:
                for n in np.arange(m + 1):
                    jsum = sum(
                        (-1)**(j + BSorder - 1) * binom(BSorder - 1, j) * (
                            (j - 0.5 * (BSorder - 2))**(BSorder + n) - (j - 0.5 * BSorder)**(BSorder + n)
                            )
                        for j in np.arange(BSorder)
                        )

                    component += (binom(m, n) * shift**(m - n) 
                                  * math.factorial(n) / math.factorial(n + BSorder) 
                                  * jsum
                                  )
            A[row, gam] = component
    b = np.zeros(R)
    b[0] = 1.0
    
    Piv = scipy.linalg.lu_factor(A)
    c_red = scipy.linalg.lu_solve(Piv, b)
    # c_red = [c_0, c_1, ..., c_RS]
    # Expand to format:
    # [c_{-RS}, ..., c_{-1}, c_0, c_1, ..., c_RS]
    cgam = np.zeros(numspline)
    
    cgam[RS] = c_red[0]
    
    for gam in range(1, R):
        cgam[RS - gam] = c_red[gam]
        cgam[RS + gam] = c_red[gam]
    
    # Sanity check: coefficients should sum to ~1 (can be outcommented)
    # sumcoeff = sum(cgam[n] for n in np.arange(numspline))
    # print('Sum of coefficients',sumcoeff) 
    return cgam

def centered_cardinal_bspline(BSorder):
    """
    Centered cardinal B-spline of given order.
    Support: [-order/2, order/2]
    Integral = 1
    """
    degree = BSorder - 1
    knots = np.arange(BSorder + 1, dtype=float)

    spline = BSpline.basis_element(knots, extrapolate=False)
    support = (-BSorder / 2, BSorder / 2)

    def B(x):
        x = np.asarray(x)
        y = spline(x + BSorder/2)
        y = np.asarray(y, dtype=float)

        mask = (x < support[0]) | (support[1] < x)
        if y.ndim == 0:
            return 0.0 if mask else float(y)
        y[mask] = 0.0
        return y

    return B
    
def grab_integrals(eval_nodes, p, BSorder, BSsupport, quad_order=None):
    """
    Compute SIAC spline-basis integrals BSInt(mode, cell, node)
    using orthonormal Legendre basis on [-1,1].

    Parameters
    ----------
    eval_nodes : array_like, shape (n_eval,)
        Reference evaluation nodes zeta_k in [-1,1].
    p : int
        DG polynomial degree.
    BSorder : int
        B-spline order.
    BSsupport : array_like of length 2
        Integer stencil bounds [min_shift, max_shift].
    quad_order : int or None
        Quadrature order for Gauss-Legendre integration.

    Returns
    -------
    BSInt : ndarray, shape (p+1, BSlen, n_eval)
        BSInt[m, j, k] = integral block for mode m,
        support-index j, evaluation-node k.
    """
    eval_nodes = np.asarray(eval_nodes, dtype=float)
    order = p + 1
    n_eval = len(eval_nodes)

    BSmin, BSmax = int(BSsupport[0]), int(BSsupport[1])
    BSlen = BSmax - BSmin + 1

    B = centered_cardinal_bspline(BSorder)

    if quad_order is None:
        quad_order = max(2 * p + 4, 12)

    q_ref, w_ref = gauss_legendre_quadrature(quad_order)

    BIntL = np.zeros((order, BSlen, n_eval))
    BIntR = np.zeros((order, BSlen, n_eval))

    for k, zeta in enumerate(eval_nodes):

        # Compute split location in reference element to separate integration
        # across B-spline piecewise regions. This aligns the quadrature with the
        # dominant breakpoint induced by the shift and spline parity.
        xicell = zeta - np.sign(zeta) * np.mod(BSorder, 2)
        # if BSorder % 2 == 0:
        #     xicell = zeta
        # else:
        #     if zeta < 0:
        #         xicell = zeta + 1.0
        #     elif zeta > 0:
        #         xicell = zeta - 1.0
        #     else:
        #         xicell = 0.0
        
        # Left interval [-1, xicell]
        qL = 0.5 * ((xicell + 1.0) * q_ref + (xicell - 1.0))
        wL = 0.5 * (xicell + 1.0) * w_ref

        # Right interval [xicell, 1]
        qR = 0.5 * ((1.0 - xicell) * q_ref + (1.0 + xicell))
        wR = 0.5 * (1.0 - xicell) * w_ref

        phiL = legendre_vandermonde(qL, p)   # (nq, order)
        phiR = legendre_vandermonde(qR, p)   # (nq, order)

        for i in range(BSmin, BSmax + 1):
            j = i - BSmin

            bsL = B(0.5 * (zeta - qL) - i)
            bsR = B(0.5 * (zeta - qR) - i)
            # Integrate B-spline values against each local basis function
            BIntL[:, j, k] = 0.5 * (phiL.T @ (wL * bsL))
            BIntR[:, j, k] = 0.5 * (phiR.T @ (wR * bsR))

    BSInt = BIntL + BIntR
    return BSInt

def apply_siac_to_modal_dg_1d(dg, moments, BSorder, points_per_cell, quad_order=None, return_blocks=False):
    """
    Apply the SIAC filter to a modal DG solution, evaluated at arbitrary
    symmetric local reference nodes in each element.

    Parameters
    ----------
    dg : dict
        DG representation with coeffs[e, mode].
    moments : int
        Number of reproduced moments. (standard 2p)
    BSorder : int
        B-spline order. (standard p+1)
    eval_nodes : array_like
        Local reference evaluation nodes in [-1,1].

    Returns
    -------
    U_star : ndarray
        SIAC field on the global grid induced by the local nodes.
        Shape = (K*n_eval).
    """  
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]
    
    p = dg["p"]
    order = p + 1
    K = mesh["K"]
    
    nodes, _ = make_evaluation_grid(mesh=mesh, points_per_cell=points_per_cell)
    
    n_eval = len(nodes)

    BSknots = np.linspace(-BSorder / 2, BSorder / 2, BSorder + 1)
    BSsupport = np.array(
        [np.floor(BSknots[0]), np.ceil(BSknots[-1])],
        dtype=int
    )
    BSlen = int(BSsupport[1] - BSsupport[0] + 1)

    cgam = siac_cgam(moments, BSorder)
    
    BSInt = grab_integrals(
        eval_nodes=nodes, 
        p=p, 
        BSorder=BSorder, 
        BSsupport=BSsupport, 
        quad_order=quad_order
    )   # (order, BSlen, n_eval)
    
    kernellength = int(2 * np.ceil((moments + BSorder) / 2) + 1)
    halfker = int(np.ceil((moments + BSorder) / 2))

    SIACmatrix = np.zeros((order, kernellength, n_eval), dtype=float)
    
    for k in range(n_eval):
        for igam in range(moments + 1):
            SIACmatrix[:, igam:igam + BSlen, k] += cgam[igam] * BSInt[:, :, k]
    
    ustar_blocks = np.zeros((K, n_eval), dtype=float)
    
    for e in range(halfker, K-halfker):
        block = coeffs[e - halfker:e + halfker + 1, :]
        
        for k in range(n_eval):
            S = SIACmatrix[:, :, k]
            # block: (kernellength, order), S: (order, kernellength)
            ustar_blocks[e, k] = np.einsum("mr,rm->", S, block)
            # ustar_blocks[e, k] = np.sum(S * block.T)

    U_star = ustar_blocks.reshape(K * n_eval)
    if return_blocks:
        return U_star, ustar_blocks
    return U_star
    