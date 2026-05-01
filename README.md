# Utilizing CLF and CBF on Adaptive Cruise Control

A reproducibility study of the Control Lyapunov Function (CLF) and Control 
Barrier Function (CBF) framework applied to Adaptive Cruise Control (ACC).

**Report:** [Utilizing CLF and CBF on Adaptive Cruise Control](AI_Robotics_fp.pdf)

**slide:** [Presentation Slides](presentation_slides.pdf)

**Author:** Wei Chien (Franklin) Kao

**Reference Paper:** [Control Barrier Function Based Quadratic Programs with Application to Adaptive Cruise Control](http://ames.caltech.edu/CLF_QP_ACC_final.pdf)  

---

## Setup

1. Create and activate a virtual environment:
```bash
   virtualenv simulation
   source simulation/bin/activate        # Linux/Mac
   simulation\Scripts\activate           # Windows
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

---

## Running the Simulation

### Static Figures (PNG)

Reproduces Figure 1 and Figure 2 from the paper:

```bash
python acc_png_production.py --case case1 --dt 0.02 --T 25.0
python acc_png_production.py --case case2 --dt 0.02 --T 25.0
```

### Animated Figures (GIF)

```bash
# Case I
python acc_gif_production.py --case case1 --dt 0.02 --T 25 --p_sc 1
python acc_gif_production.py --case case1 --dt 0.02 --T 25 --p_sc 1e-5

# Case II
python acc_gif_production.py --case case2 --dt 0.02 --T 25 --p_sc 1
python acc_gif_production.py --case case2 --dt 0.02 --T 25 --p_sc 1e-5
```

One can also access the simulation result at simulation_result/.

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--case` | Simulation case (`case1` or `case2`) | required |
| `--dt` | Time step (seconds) | `0.02` |
| `--T` | Simulation duration (seconds) | `25.0` |
| `--p_sc` | CLF relaxation penalty weight | `1e-5` |

---

## Cases

- **Case 1:** CLF + CBF constraints only
- **Case 2:** CLF + CBF + Force-based CBF (FCBF) + Control Constraints (CC)