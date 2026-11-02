"""Step 5 - Compare the real arm with the LSTM simulator for the same voltage sequence.

Replays the motor-voltage CSV that was sent to the robot through the LSTM
simulator (starting from rest) and overlays the encoder / strain-gauge log
recorded on the real arm.

Usage:
    python src/compare_sim_vs_real.py
    python src/compare_sim_vs_real.py --sequence results/generated_action_sequence.csv --real robot_output_xxx.csv
"""
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config as C
from robot_lstm import RobotLSTM, simulate_robot_dynamics

EXP_DIR = os.path.join(C.ROOT_DIR, "experiments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=os.path.join(EXP_DIR, "generated_action_sequence.csv"))
    parser.add_argument("--real", default=os.path.join(EXP_DIR, "real_run_rl_output.csv"))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    seq = pd.read_csv(args.sequence)
    real = pd.read_csv(args.real)
    mv_col = "Motor Voltage (V)" if "Motor Voltage (V)" in real.columns else "Commanded Motor Voltage (V)"
    real = real[real["Strain Gauge Voltage (V)"] > 0.5].copy()   # drop the first sample before the ADC is ready
    # The logger starts before the first command: align t = 0 with the first non-zero command
    t0 = real.loc[real[mv_col].abs() > 1e-6, "Time (s)"].min()
    real["Time (s)"] -= t0 if np.isfinite(t0) else 0.0
    real = real[real["Time (s)"] >= -0.05]

    model = RobotLSTM(C.INPUT_SIZE, C.HIDDEN_SIZE, C.OUTPUT_SIZE, num_layers=C.NUM_LSTM_LAYERS)
    model.load_state_dict(torch.load(C.LSTM_MODEL_PATH, map_location="cpu"))
    scaler_X, scaler_y = joblib.load(C.SCALER_X_PATH), joblib.load(C.SCALER_Y_PATH)

    init = np.zeros((C.LOOK_BACK_STEPS, 3))
    init[:, 2] = C.STRAIN_OFFSET
    sim = simulate_robot_dynamics(model, init, seq["Motor Voltage (V)"].to_numpy(), scaler_X, scaler_y, "cpu")
    t_sim = (np.arange(len(sim)) + 1) / C.SAMPLING_RATE_HZ

    tail = real["Time (s)"] > real["Time (s)"].max() - 1.0
    print(f"Simulated final position : {sim[-1, 0]:.2f} deg")
    print(f"Real arm, last 1 s mean  : {real.loc[tail, 'Encoder Position (Degrees)'].mean():.2f} deg "
          f"(target {C.TARGET_POSITION:.0f} deg)")

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax[0].step(seq["Time (s)"], seq["Motor Voltage (V)"], where="post", color="black", label="sequence (from SAC)")
    ax[0].plot(real["Time (s)"], real[mv_col], color="orange", alpha=0.8, label="commanded on robot")
    ax[0].set_ylabel("Motor voltage (V)")
    ax[1].plot(t_sim, sim[:, 0], color="blue", label="LSTM simulation")
    ax[1].plot(real["Time (s)"], real["Encoder Position (Degrees)"], color="tab:cyan", lw=2, label="real arm")
    ax[1].axhline(C.TARGET_POSITION, ls="--", color="green", label="target")
    ax[1].set_ylabel("Encoder position (deg)")
    ax[2].plot(t_sim, sim[:, 1], color="red", label="LSTM simulation")
    ax[2].plot(real["Time (s)"], real["Strain Gauge Voltage (V)"], color="salmon", lw=1.5, label="real arm")
    ax[2].axhline(C.STRAIN_OFFSET, ls="--", color="purple", label="rest offset")
    ax[2].set_ylabel("Strain gauge (V)")
    ax[2].set_xlabel("Time (s)")
    for a in ax:
        a.grid(True)
        a.legend(loc="upper right")
    fig.suptitle("RL voltage sequence: LSTM simulation vs. real flexible arm")
    fig.tight_layout()

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = os.path.join(C.RESULTS_DIR, "sim_vs_real.png")
    fig.savefig(out, dpi=120)
    print(f"Saved plot -> {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
