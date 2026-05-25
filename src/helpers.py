
import numpy as np
import pandas as pd

from src.mesh import build_uniform_mesh
from src.projection import l2_project_mesh, l2_project_mesh_noisy
from src.dg_solver import dg_rhs_advection
from src.time_integrators import forward_euler_step, rk4_step, time_integrator
from src.cfl import estimate_dt_advection, choose_time_step
from src.evaluation import evaluate_dg_solution
from src.siac import apply_siac_to_modal_dg_1d

def run_dg_siac_advection_experiment(
    K,
    p,
    a,
    T,
    domain,
    exact_func,
    time_method="rk4",
    ppe=20,
    cfl_safety=0.9,
    moments=None,
    BSorder=None,
    quad_order=None,
    time_tol=1e-12,
):
    """
    Run one DG + SIAC experiment for the 1D linear advection equation.

    Parameters
    ----------
    K : int
        Number of mesh elements.
    p : int
        DG polynomial degree.
    a : float
        Advection speed.
    T : float
        Final time.
    domain : tuple
        Physical domain (xmin, xmax).
    exact_func : callable
        Initial condition u0(x).
    time_method : str
        Time integrator, e.g. "rk4" or "euler".
    ppe : int
        Points per element for evaluation.
    cfl_safety : float
        Safety factor for CFL-based timestep.
    moments : int or None
        SIAC reproduction degree/moments. If None, use 2*p.
    BSorder : int or None
        B-spline order. If None, use p+1.
    quad_order : int or None
        Quadrature order used inside SIAC.
    time_tol : float
        Time-roundoff tolerance.

    Returns
    -------
    result : dict
        Dictionary containing DG solution, SIAC solution, exact solution,
        grids, and time-step information.
    """
    if moments is None:
        moments = 2 * p

    if BSorder is None:
        BSorder = p + 1

    # Mesh
    mesh = build_uniform_mesh(K=K, domain=domain)

    # Initial DG projection
    dg = l2_project_mesh(
        f=exact_func,
        p=p,
        mesh=mesh,
        num_quad=None,
    )

    # Exact solution of u_t + a u_x = 0
    def exact_solution(x, t):
        return exact_func(x - a * t)

    # RHS wrapper
    def rhs(u, t):
        return dg_rhs_advection(
            coeffs=u,
            mesh=mesh,
            p=p,
            a=a,
        )

    # Time step
    dt_est = estimate_dt_advection(
        mesh=mesh,
        p=p,
        a=a,
        method=time_method,
        safety=cfl_safety,
    )

    dt, num_steps = choose_time_step(T=T, dt_est=dt_est)

    # Time stepping
    u = dg["coeffs"].copy()
    t = 0.0

    for n in range(num_steps):
        dt_step = min(dt, T - t)

        if dt_step <= time_tol:
            break

        u = time_integrator(rhs, u, t, dt_step, method=time_method)
        t += dt_step

    # Snap final time if only roundoff remains
    if abs(T - t) <= time_tol:
        t = T
    else:
        raise RuntimeError(f"Time stepping stopped at t={t}, not T={T}.")

    # Final DG dictionary
    dg_final = dg.copy()
    dg_final["coeffs"] = u
    dg_final["t"] = t

    # Evaluation
    eval_grid, U_dg = evaluate_dg_solution(
        dg_final,
        points_per_cell=ppe,
    )

    U_exact = exact_solution(eval_grid, t)

    # SIAC postprocessing
    U_star = apply_siac_to_modal_dg_1d(
        dg=dg_final,
        moments=moments,
        BSorder=BSorder,
        points_per_cell=ppe,
        quad_order=quad_order,
    )

    # Interior slice where symmetric SIAC is valid
    halfker = int(np.ceil((moments + BSorder) / 2))

    start = halfker * ppe
    stop = (K - halfker) * ppe

    eval_grid_int = eval_grid[start:stop]
    U_dg_int = U_dg[start:stop]
    U_exact_int = U_exact[start:stop]
    U_star_int = U_star[start:stop]

    result = {
        "K": K,
        "p": p,
        "a": a,
        "T": T,
        "domain": domain,
        "mesh": mesh,
        "dg_initial": dg,
        "dg_final": dg_final,
        "eval_grid": eval_grid,
        "U_dg": U_dg,
        "U_exact": U_exact,
        "U_star": U_star,
        "eval_grid_int": eval_grid_int,
        "U_dg_int": U_dg_int,
        "U_exact_int": U_exact_int,
        "U_star_int": U_star_int,
        "dt": dt,
        "dt_est": dt_est,
        "num_steps": num_steps,
        "t_final": t,
        "time_method": time_method,
        "ppe": ppe,
        "moments": moments,
        "BSorder": BSorder,
        "halfker": halfker,
    }

    return result

def run_dg_siac_noise_experiment(
    *,
    exact_func,
    exact_solution=None,
    a=2.0,
    T=2.0,
    time_method="rk4",
    domain=(-1.0, 1.0),
    p=2,
    K=32,
    q=None,
    C=1.0,
    seed=None,
    num_quad_projection=None,
    ppe=20,
    moments=None,
    BSorder=None,
    cfl_safety=0.9,
    time_tol=1e-12,
):
    """
    Run one DG advection + noisy initial projection + SIAC postprocessing experiment.

    This function performs no printing and no error calculations. It returns the
    DG solution, SIAC solution, exact values, grids, and metadata so that errors
    can be computed externally.

    Parameters
    ----------
    exact_func : callable
        Initial condition f(x).

    exact_solution : callable or None
        Exact time-dependent solution exact_solution(x, t). If None, assumes
        periodic linear advection:
            exact_solution(x, t) = exact_func(x - a*t)

    a : float
        Constant advection speed.

    T : float
        Final time.

    time_method : str
        Time integration method, e.g. "rk4" or "forward_euler".

    domain : tuple
        Domain (xmin, xmax).

    p : int
        DG polynomial degree.

    K : int
        Number of elements.

    q : int or float
        Noise scaling exponent. Noise amplitude is epsilon = C h^q.
        If None, defaults to p + 1.

    C : float
        Noise scaling constant.

    seed : int or None
        Random seed for noisy projection.

    num_quad_projection : int or None
        Number of quadrature points used in the noisy L2 projection.

    ppe : int
        Points per element for evaluating DG and SIAC solutions.

    moments : int or None
        SIAC moment/reproduction parameter. If None, defaults to 2*p.

    BSorder : int or None
        B-spline order used in SIAC. If None, defaults to p + 1.

    cfl_safety : float
        Safety factor used in the time-step estimate.

    time_tol : float
        Tolerance for final-time snapping.

    Returns
    -------
    result : dict
        Dictionary containing final DG solution, evaluated arrays, SIAC solution,
        interior slices, and metadata.
    """

    if q is None:
        q = p + 1

    if moments is None:
        moments = 2 * p

    if BSorder is None:
        BSorder = p + 1

    order = p + 1

    # Build mesh
    mesh = build_uniform_mesh(K=K, domain=domain)

    # Project noisy initial condition
    dg_initial = l2_project_mesh_noisy(
        f=exact_func,
        p=p,
        mesh=mesh,
        q=q,
        C=C,
        seed=seed,
        num_quad=num_quad_projection,
    )

    # RHS wrapper
    def rhs(u, t):
        return dg_rhs_advection(
            coeffs=u,
            mesh=mesh,
            p=p,
            a=a,
        )

    # Choose stable time step
    dt_est = estimate_dt_advection(
        mesh=mesh,
        p=p,
        a=a,
        method=time_method,
        safety=cfl_safety,
    )
    dt, num_steps = choose_time_step(T=T, dt_est=dt_est)

    # Time stepping
    u = dg_initial["coeffs"].copy()
    t = 0.0

    for _ in range(num_steps):
        dt_step = min(dt, T - t)

        if dt_step <= time_tol:
            break

        u = time_integrator(rhs, u, t, dt_step, method=time_method)
        t += dt_step

    # Snap final time to T if only roundoff remains
    if abs(T - t) <= time_tol:
        t = T
    else:
        raise RuntimeError(f"Time stepping stopped at t={t}, not T={T}.")

    # Store final DG state
    dg_final = dg_initial.copy()
    dg_final["coeffs"] = u
    dg_final["t"] = t

    # Evaluation grid and DG solution
    eval_grid, U_dg = evaluate_dg_solution(
        dg_final,
        points_per_cell=ppe,
    )

    # Exact solution
    if exact_solution is None:
        def exact_solution(x, t):
            return exact_func(x - a * t)

    U_exact = exact_solution(eval_grid, t)

    # SIAC postprocessing
    U_star = apply_siac_to_modal_dg_1d(
        dg=dg_final,
        moments=moments,
        BSorder=BSorder,
        points_per_cell=ppe,
    )

    # Interior slice where SIAC is valid
    halfker = int(np.ceil((moments + BSorder) / 2))

    start = halfker * ppe
    stop = (K - halfker) * ppe

    if stop <= start:
        raise ValueError(
            "Interior SIAC comparison region is empty. "
            f"Got K={K}, halfker={halfker}, ppe={ppe}. "
            "Increase K or reduce the SIAC stencil size."
        )

    eval_grid_int = eval_grid[start:stop]
    U_dg_int = U_dg[start:stop]
    U_exact_int = U_exact[start:stop]
    U_star_int = U_star[start:stop]

    result = {
        # DG objects
        "dg_initial": dg_initial,
        "dg_final": dg_final,

        # Full evaluated arrays
        "eval_grid": eval_grid,
        "U_dg": U_dg,
        "U_exact": U_exact,
        "U_star": U_star,

        # Interior arrays for valid SIAC comparison
        "eval_grid_int": eval_grid_int,
        "U_dg_int": U_dg_int,
        "U_exact_int": U_exact_int,
        "U_star_int": U_star_int,

        # Slicing metadata
        "interior_slice": slice(start, stop),
        "halfker": halfker,
        "start": start,
        "stop": stop,

        # Time metadata
        "t_final": t,
        "dt": dt,
        "dt_est": dt_est,
        "num_steps": num_steps,

        # Experiment metadata
        "params": {
            "a": a,
            "T": T,
            "time_method": time_method,
            "domain": domain,
            "p": p,
            "order": order,
            "K": K,
            "q": q,
            "C": C,
            "seed": seed,
            "num_quad_projection": num_quad_projection,
            "ppe": ppe,
            "moments": moments,
            "BSorder": BSorder,
            "cfl_safety": cfl_safety,
        },
    }

    return result