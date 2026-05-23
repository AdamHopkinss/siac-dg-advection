import numpy as np


CFL_TABLE = {
    # Conservative safe defaults for 1D DG advection with upwind flux.
    # dt = cfl * h / ((2*p + 1) * abs(a))
    "forward_euler": 0.5,
    "euler": 0.5,
    "rk4": 1.0,
}


def estimate_dt_advection(mesh, p, a, method="rk4", safety=0.9):
    """
    Estimate stable time step for 1D DG linear advection.

    Uses the heuristic CFL condition

        dt <= CFL * h / ((2*p + 1) * |a|).

    Parameters
    ----------
    mesh : dict
        Mesh dictionary containing "h".
    p : int
        DG polynomial degree.
    a : float
        Advection speed.
    method : str
        Time integration method, e.g. "rk4" or "forward_euler".
    safety : float
        Extra safety factor.

    Returns
    -------
    dt : float
        Suggested time step.
    """
    if a == 0:
        return np.inf

    method = method.lower()

    if method not in CFL_TABLE:
        raise ValueError(f"Unknown method '{method}'. Available: {list(CFL_TABLE)}")

    cfl = CFL_TABLE[method]

    if np.isinf(cfl):
        return np.inf

    h = mesh["h"]

    # If h is an array, use the smallest cell.
    h_min = np.min(h)

    dt = safety * cfl * h_min / ((2*p + 1) * abs(a))

    return dt

def choose_time_step(T, dt_est):
    """
    Adjust dt so that an integer number of time steps reaches T exactly.

    Parameters
    ----------
    T : float
        Final time.
    dt_est : float
        Estimated stable time step.

    Returns
    -------
    dt : float
        Adjusted time step.
    num_steps : int
        Number of steps.
    """
    if np.isinf(dt_est):
        raise ValueError("Cannot choose time step from infinite dt estimate.")

    num_steps = int(np.ceil(T / dt_est))
    dt = T / num_steps

    return dt, num_steps