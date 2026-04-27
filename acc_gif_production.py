
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_case_module(case_name):
    case_name = case_name.lower()
    module_map = {
        "case1": "simulation_case1",
        "case2": "simulation_case2",
    }
    if case_name not in module_map:
        raise ValueError("case_name must be 'case1' or 'case2'.")
    return importlib.import_module(module_map[case_name])


def animate_acc(case_name="case1", dt=0.01, T=20.0, p_sc=1e-5, save_gif=True):
    case_module = load_case_module(case_name)
    params = case_module.params

    # run the simulation
    params["p_sc"] = p_sc
    t, x2_traj, u_traj, h_traj = case_module.euler_simulate(dt=dt, T=T)

    # Reconstruct positions from velocity
    ego_pos = np.zeros_like(t)
    lead_pos = np.zeros_like(t)

    ego_pos[0] = 0.0
    lead_pos[0] = 100.0  # same initial z

    for i in range(1, len(t)):
        ego_pos[i] = ego_pos[i - 1] + x2_traj[i - 1] * dt
        lead_pos[i] = lead_pos[i - 1] + params["v_0"] * dt

    z_traj = lead_pos - ego_pos

    fig, axes = plt.subplots(4, 1, figsize=(10, 9))
    ax_road, ax_speed, ax_u, ax_h = axes

    # ---------------- Road view ----------------
    ax_road.set_title(f"ACC Animation ({case_name}), p_sc = {p_sc}")
    ax_road.set_ylim(-1, 1)
    ax_road.set_yticks([])
    ax_road.set_xlabel("Position (m)")

    road_line, = ax_road.plot([], [], "k--", alpha=0.3)
    ego_dot, = ax_road.plot([], [], "bo", markersize=14, label="Ego car")
    lead_dot, = ax_road.plot([], [], "ro", markersize=14, label="Lead car")
    safe_line, = ax_road.plot([], [], "g--", label="Safety distance boundary")

    ax_road.legend(loc="upper right")

    # ---------------- Speed plot ----------------
    ax_speed.set_title("Ego Speed")
    ax_speed.set_xlim(0, T)
    ax_speed.set_ylim(min(x2_traj) - 2, max(params["v_d"], max(x2_traj)) + 2)
    ax_speed.set_ylabel("m/s")
    speed_line, = ax_speed.plot([], [], label="ego speed")
    ax_speed.axhline(params["v_d"], linestyle="--", color="k", label="desired speed")
    ax_speed.axhline(params["v_0"], linestyle="--", color="gray", label="lead speed")
    ax_speed.legend()

    # ---------------- Control input plot ----------------
    ax_u.set_title("Control Input")
    ax_u.set_xlim(0, T)
    ax_u.set_ylim(min(u_traj) - 0.05, max(u_traj) + 0.05)
    ax_u.set_ylabel("u / mg")
    u_line, = ax_u.plot([], [], label="u / mg")
    ax_u.axhline(params["c_alpha"], linestyle="--", color="k")
    ax_u.axhline(-params["c_d"], linestyle="--", color="k")
    ax_u.legend()

    # ---------------- Safety h plot ----------------
    ax_h.set_title("Safety Constraint h(x,z)")
    ax_h.set_xlim(0, T)
    ax_h.set_ylim(min(h_traj) - 5, max(h_traj) + 5)
    ax_h.set_xlabel("Time (s)")
    ax_h.set_ylabel("h")
    h_line, = ax_h.plot([], [], label="h(x,z)")
    ax_h.axhline(0, linestyle="--", color="k", label="safety boundary")
    ax_h.legend()

    skip = 5  # animate every 5 simulation steps

    def init():
        road_line.set_data([], [])
        ego_dot.set_data([], [])
        lead_dot.set_data([], [])
        safe_line.set_data([], [])
        speed_line.set_data([], [])
        u_line.set_data([], [])
        h_line.set_data([], [])
        return road_line, ego_dot, lead_dot, safe_line, speed_line, u_line, h_line

    def update(frame):
        i = frame * skip
        if i >= len(t):
            i = len(t) - 1

        ego_x = ego_pos[i]
        lead_x = lead_pos[i]

        window_left = ego_x - 30
        window_right = lead_x + 50

        ax_road.set_xlim(window_left, window_right)

        road_line.set_data([window_left, window_right], [0, 0])
        ego_dot.set_data([ego_x], [0])
        lead_dot.set_data([lead_x], [0])

        # Safety distance boundary: lead position minus required distance
        safe_boundary = lead_x - 1.8 * x2_traj[i]
        safe_line.set_data([safe_boundary, safe_boundary], [-0.5, 0.5])

        speed_line.set_data(t[:i], x2_traj[:i])
        u_line.set_data(t[:i], u_traj[:i])
        h_line.set_data(t[:i], h_traj[:i])

        return road_line, ego_dot, lead_dot, safe_line, speed_line, u_line, h_line

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t) // skip,
        init_func=init,
        interval=30,
        blit=False
    )

    plt.tight_layout()

    if save_gif:
        filename = f"acc_animation_{case_name}_psc_{p_sc}.gif"
        anim.save(filename, writer=PillowWriter(fps=60)) # change to 60 frames per seconds
        print(f"Saved animation to {filename}")

    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACC GIF for case1/case2.")
    parser.add_argument("--case", choices=["case1", "case2"], default="case1")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--p_sc", type=float, default=1.0)
    parser.add_argument("--no-save", action="store_true", help="Do not save GIF file")

    args = parser.parse_args()

    animate_acc(
        case_name=args.case,
        dt=args.dt,
        T=args.T,
        p_sc=args.p_sc,
        save_gif=not args.no_save,
    )