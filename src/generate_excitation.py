"""Step 0 - Generate excitation voltage profiles for data collection.

The LSTM was trained on 520 runs of 6 s each, excited with step, pulse and
sine inputs. This script writes such profiles in the same CSV format that
raspberry_pi/run_on_robot.py plays back, so the logs it records can be used
directly as training data (put them in data/runs/).

Every profile starts with 2 s at 0 V (this part is cut off during training).

Usage:
    python src/generate_excitation.py --out data/excitation --count 50
"""
import argparse
import csv
import os

import numpy as np

import config as C

FS = C.SAMPLING_RATE_HZ
LEAD_IN_S = C.DATA_START_TIME_S
RUN_LENGTH_S = 6.0


def step_profile(rng, n):
    u = np.zeros(n)
    start = rng.integers(0, n // 3)
    u[start:] = rng.uniform(-0.6, 0.6)
    return u


def pulse_profile(rng, n):
    u = np.zeros(n)
    start = rng.integers(0, n // 3)
    width = int(rng.uniform(0.1, 0.8) * FS)
    u[start:start + width] = rng.choice([-1, 1]) * rng.uniform(0.15, 1.0)
    return u


def sine_profile(rng, n):
    t = np.arange(n) / FS
    return rng.uniform(0.1, 0.6) * np.sin(2 * np.pi * rng.uniform(0.2, 4.0) * t + rng.uniform(0, 2 * np.pi))


PROFILES = {"step": step_profile, "pulse": pulse_profile, "sine": sine_profile}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(C.ROOT_DIR, "data", "excitation"))
    parser.add_argument("--count", type=int, default=30, help="profiles per type")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_lead, n_run = int(LEAD_IN_S * FS), int((RUN_LENGTH_S - LEAD_IN_S) * FS)
    for kind, fn in PROFILES.items():
        for i in range(args.count):
            u = np.clip(np.concatenate([np.zeros(n_lead), fn(rng, n_run)]), C.MOTOR_VOLTAGE_MIN, C.MOTOR_VOLTAGE_MAX)
            path = os.path.join(args.out, f"{kind}_{i:03d}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Time (s)", "Motor Voltage (V)"])
                for k, v in enumerate(u):
                    w.writerow([round(k / FS, 4), round(float(v), 4)])
    print(f"Wrote {len(PROFILES) * args.count} profiles to {args.out}")


if __name__ == "__main__":
    main()
