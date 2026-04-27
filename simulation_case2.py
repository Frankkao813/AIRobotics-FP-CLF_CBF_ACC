import math
import numpy as np

from qpsolvers import solve_qp


params = {
    "g": 9.81,
    "f_0": 0.1,
    "f_1": 5.0,
    "f_2": 0.25,
    "v_d": 24.0,
    "delta_sc": 0.0,
    "m": 1650.0,
    "epsilon": 10.0,
    "v_0": 13.89,
    "gamma": 1.0,
    "c_d": 0.3,
    "c_alpha": 0.3,
    "p_sc": 1e-5
}


# --------------------------------------------------
# Basic helper
# --------------------------------------------------
def rolling_resistance(v):
    return params["f_0"] + params["f_1"] * v + params["f_2"] * v**2


# --------------------------------------------------
# Soft constraint as CLF
# Paper:
# psi_0(x,z) = -(2(x2-vd)/m) * Fr(x) + epsilon * (x2-vd)^2
# psi_1(x,z) =  2(x2-vd)/m
# --------------------------------------------------
def psi_0(x, z):
    x2 = x[1]
    y = x2 - params["v_d"]
    Fr = rolling_resistance(x2)
    return -(2.0 * y / params["m"]) * Fr + params["epsilon"] * (y ** 2)


def psi_1(x, z):
    x2 = x[1]
    return 2.0 * (x2 - params["v_d"]) / params["m"]


# --------------------------------------------------
# Hard constraint HC1 as CBF
# h(x,z) = z - 1.8 x2
# B(x,z) = -log(h / (1+h))
# --------------------------------------------------
def h(x, z):
    return z - 1.8 * x[1]


def barrier_function(x, z):
    hx = h(x, z)
    if hx <= 0:
        raise ValueError("h(x,z) must be positive for the log barrier.")
    return -math.log(hx / (1.0 + hx))


def L_fh(x, z):
    """
    h(x,z) = z - 1.8 x2

    Full drift:
      x1_dot = x2
      x2_dot = -Fr/m
      z_dot  = v0 - x2

    So:
      L_f h = [0, -1.8, 1] dot [x2, -Fr/m, v0-x2]
            = 1.8*Fr/m + (v0-x2)
    """
    x2 = x[1]
    Fr = rolling_resistance(x2)
    return 1.8 * Fr / params["m"] + (params["v_0"] - x2)


def L_gh(x, z):
    """
    g_bar = [0, 1/m, 0]^T
    grad h = [0, -1.8, 1]^T

    L_g h = grad h dot g_bar = -1.8 / m
    """
    return -1.8 / params["m"]


def L_fB(x, z):
    """
    B = -log(h/(1+h))
    dB/dh = -1 / (h(1+h))
    So L_f B = (dB/dh) * L_f h
    """
    hx = h(x, z)
    return -(1.0 / (hx * (1.0 + hx))) * L_fh(x, z)


def L_gB(x, z):
    """
    B = -log(h/(1+h))
    dB/dh = -1 / (h(1+h))
    So L_g B = (dB/dh) * L_g h
    """
    hx = h(x, z)
    return -(1.0 / (hx * (1.0 + hx))) * L_gh(x, z)


def HC1_CBF(x, z, u):
    B = barrier_function(x, z)
    return L_fB(x, z) + L_gB(x, z) * u - params["gamma"] / B


# --------------------------------------------------
# Force-based constraint
# Paper:
# h_F(x,z) = z - 1.8 x2 - (v0 - x2)^2 / (2 c_d g)
# B_F(x,z) = 1 / h_F(x,z)
# --------------------------------------------------
def h_f(x, z):
    x2 = x[1]
    return z - 1.8 * x2 - ((params["v_0"] - x2) ** 2) / (
        2.0 * params["c_d"] * params["g"]
    )


def B_f(x, z):
    hf = h_f(x, z)
    if hf <= 0:
        raise ValueError("h_F(x,z) must be positive for B_F = 1/h_F.")
    return 1.0 / hf


def L_fh_f(x, z):
    """
    h_F(x,z) = z - 1.8 x2 - (v0-x2)^2 / (2 c_d g)

    grad h_F wrt [x1, x2, z]:
      [0,
       -1.8 + (v0-x2)/(c_d g),
       1]

    full drift:
      [x2, -Fr/m, v0-x2]

    Therefore:
      L_f h_F
      = (-1.8 + (v0-x2)/(c_d g)) * (-Fr/m) + (v0-x2)
    """
    x2 = x[1]
    Fr = rolling_resistance(x2)

    coeff_x2 = -1.8 + (params["v_0"] - x2) / (params["c_d"] * params["g"])
    x2_drift = -Fr / params["m"]
    z_drift = params["v_0"] - x2

    return coeff_x2 * x2_drift + z_drift


def L_gh_f(x, z):
    """
    grad h_F dot g_bar
    = (-1.8 + (v0-x2)/(c_d g)) * (1/m)
    """
    x2 = x[1]
    coeff_x2 = -1.8 + (params["v_0"] - x2) / (params["c_d"] * params["g"])
    return coeff_x2 / params["m"]


def L_fB_f(x, z):
    hf = h_f(x, z)
    return -(1.0 / (hf ** 2)) * L_fh_f(x, z)


def L_gB_f(x, z):
    hf = h_f(x, z)
    return -(1.0 / (hf ** 2)) * L_gh_f(x, z)


def force_based_constraint(x, z, u):
    return L_fB_f(x, z) + L_gB_f(x, z) * u - 1.0 / B_f(x, z)




def compute_H_acc(m, p_sc):
    return 2 * np.array([
        [1 / m**2, 0],
        [0, p_sc]
    ])


def compute_F_acc(F_r, m):
    return -2 * np.array([F_r / m**2, 0.0])

def compute_A_cc():
    return np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])


def compute_b_cc():
    return np.array([
        params["c_alpha"] * params["m"] * params["g"],
        params["c_d"] * params["m"] * params["g"]
    ])


def solve_acc_qp(x, z, params):
    """
    Solve the ACC QP given state x, z, and system parameters.

    Parameters
    ----------
    x      : np.ndarray - state vector
    z      : float      - auxiliary state
    params : dict       - system parameters (m, p_sc, gamma, c_a, c_d, g, etc.)

    Returns
    -------
    u_opt      : float - optimal control input
    delta_sc   : float - optimal slack variable
    """
    m = params["m"]
    p_sc = params["p_sc"]
    gamma = params["gamma"]

    F_r = rolling_resistance(x[1])

    # Cost
    H = compute_H_acc(m, p_sc)
    F = compute_F_acc(F_r, m)

    # Constraints: stack A*u <= b
    A = np.vstack([
        [psi_1(x, z), -1.0],
        [L_gB(x, z), 0.0],
        [L_gB_f(x, z), 0.0],
        compute_A_cc()
    ])

    b = np.concatenate([
        [-psi_0(x, z)],
        [gamma / barrier_function(x, z) - L_fB(x, z)],
        [1.0 / B_f(x, z) - L_fB_f(x, z)],
        compute_b_cc()
    ])

    solution = solve_qp(H, F, G=A, h=b, solver="osqp")

    if solution is None:
        raise RuntimeError("QP solver failed to find a solution.")

    u_opt, delta_sc = solution
    return u_opt, delta_sc






# dynamics simulation
def dynamics(x, z, u):
    x2 = x[1]
    Fr = rolling_resistance(x2)

    x1_dot = x2
    x2_dot = (u - Fr) / params["m"]
    z_dot = params["v_0"] - x2

    return np.array([x1_dot, x2_dot]), z_dot



def euler_simulate(dt, T):

    steps = int(T / dt)

    x = np.array([0.0, 20.0])
    z = 100.0   # same as paper initial condition

    x2_traj = []
    u_traj = []
    h_traj = []
    t_traj = []

    for i in range(steps):
        t = i * dt

        u, _ = solve_acc_qp(x, z, params)

        x_dot, z_dot = dynamics(x, z, u)

        # Euler integration
        x = x + dt * x_dot
        z = z + dt * z_dot

        # store
        x2_traj.append(x[1])
        u_traj.append(u / (params["m"] * params["g"]))  # normalized like paper
        h_traj.append(h(x, z))
        t_traj.append(t)
    return np.array(t_traj), np.array(x2_traj), np.array(u_traj), np.array(h_traj)









