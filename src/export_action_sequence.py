"""Step 4 - Export the trained policy as a motor-voltage sequence for the real arm.

The Raspberry Pi 4 was not fast enough to run the LSTM + SAC policy in real
time, so the agent is rolled out in the LSTM simulator from the arm's rest
state and the resulting voltage profile is saved as a CSV. That file is
copied to the Pi and played back by raspberry_pi/run_on_robot.py.

Usage:
    python src/export_action_sequence.py                 # uses models/final_robot_sac_agent.zip
    python src/export_action_sequence.py --best          # uses the EvalCallback best model
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import SAC

import config as C
from robot_environment import RobotEnv
from train_sac_agent import plot_episode, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=C.SAC_AGENT_PATH + ".zip")
    parser.add_argument("--best", action="store_true", help="use logs/best_rl_model/best_model.zip")
    parser.add_argument("--output", default=C.ACTION_SEQUENCE_CSV)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    agent_path = os.path.join(C.SAC_BEST_AGENT_DIR, "best_model.zip") if args.best else args.agent

    # Start exactly from rest: encoder reset to 0 deg, link not vibrating.
    env = RobotEnv(initial_pos_range=(0.0, 0.0), initial_strain_offset_range=(0.0, 0.0))
    # custom_objects: only the policy is needed, so skip the training schedules
    # (avoids unpickling errors when the Python version differs from training)
    model = SAC.load(agent_path, device="cpu",
                     custom_objects={"lr_schedule": lambda _: 3e-4, "learning_rate": 3e-4})
    t, eps, sgvs, mvs, total = run_episode(model, env)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time (s)", "Motor Voltage (V)"])
        for k, mv in enumerate(mvs):
            w.writerow([round(k * env.time_step_delta, 4), round(mv, 4)])
    print(f"Wrote {len(mvs)} commands ({len(mvs) * env.time_step_delta:.2f} s) -> {args.output}")
    print(f"Simulated final position {eps[-1]:.2f} deg, final strain {sgvs[-1]:.4f} V, reward {total:.1f}")

    plot_episode(t, eps, sgvs, mvs, os.path.join(C.RESULTS_DIR, "exported_sequence_simulation.png"),
                 not args.no_show)


if __name__ == "__main__":
    main()
