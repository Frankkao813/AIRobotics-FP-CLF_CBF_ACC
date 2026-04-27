
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt





def load_case_module(case_name):
    case_name = case_name.lower()
    module_map = {
        "case1": "simulation_case1",
        "case2": "simulation_case2",
    }
    if case_name not in module_map:
        raise ValueError("case_name must be 'case1' or 'case2'.")
    return importlib.import_module(module_map[case_name])


def png_acc(args):
    case_module = load_case_module(args.case)
    params = case_module.params

    # --- Run two simulations ---
    params["p_sc"] = 1e-5
    t1, x2_1, u1, h1 = case_module.euler_simulate(dt=args.dt, T=args.T)

    params["p_sc"] = 1.0
    t2, x2_2, u2, h2 = case_module.euler_simulate(dt=args.dt, T=args.T)

    plt.figure(figsize=(15,4))
    plt.suptitle(f"ACC {args.case.upper()} Comparison (p_sc = 1e-5 vs p_sc = 1)", fontsize=14)

    # --- Speed ---
    plt.subplot(1,3,1)
    plt.plot(t1, x2_1, 'r', label="p_sc = 1e-5")
    plt.plot(t2, x2_2, 'b', label="p_sc = 1")
    plt.axhline(params["v_d"], linestyle='--', color='k', label="v_d")
    plt.axhline(params["v_0"], linestyle='--', color='gray', label="v_0")
    plt.title("Speed (m/s)")
    plt.xlabel("t")
    plt.legend()

    # --- Wheel force ---
    plt.subplot(1,3,2)
    plt.plot(t1, u1, 'r', label="p_sc = 1e-5")
    plt.plot(t2, u2, 'b', label="p_sc = 1")
    plt.axhline(params["c_alpha"], linestyle='--', color='k')
    plt.axhline(-params["c_d"], linestyle='--', color='k')
    plt.title("Wheel Force (u/mg)")
    plt.xlabel("t")
    plt.legend()

    # --- Constraint ---
    plt.subplot(1,3,3)
    plt.plot(t1, h1, 'r', label="p_sc = 1e-5")
    plt.plot(t2, h2, 'b', label="p_sc = 1")
    plt.axhline(0, linestyle='--', color='k')
    plt.title("h(x)")
    plt.xlabel("t")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"acc_simulation_compare_{args.case}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACC PNG for case1/case2.")
    parser.add_argument("--case", help="Case to simulate", choices=["case1", "case2"], default="case1")
    parser.add_argument("--dt", help="Time step", type=float, default=0.01)
    parser.add_argument("--T", help="Total simulation time", type=float, default=20.0)
    args = parser.parse_args()
    png_acc(args)

 