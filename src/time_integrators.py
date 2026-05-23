

def forward_euler_step(rhs, u, t, dt):
    return u + dt * rhs(u, t)


def rk4_step(rhs, u, t, dt):
    k1 = rhs(u, t)
    k2 = rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(u + dt * k3, t + dt)

    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def time_integrator(rhs, u, t, dt, method="rk4"):
    method = method.lower()

    methods = {
        "rk4": rk4_step,
        "euler": forward_euler_step,
        "forward_euler": forward_euler_step,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown time integration method '{method}'. "
            f"Available methods are: {list(methods.keys())}"
        )

    return methods[method](rhs, u, t, dt)