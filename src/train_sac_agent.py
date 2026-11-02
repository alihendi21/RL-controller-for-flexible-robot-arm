"""Step 3 - Train the Soft Actor-Critic (SAC) agent in the LSTM-based environment.

Usage:
    python src/train_sac_agent.py                       # 600k steps (~3 h on a laptop GPU)
    python src/train_sac_agent.py --timesteps 20000     # quick test
TensorBoard:
    tensorboard --logdir logs/tensorboard
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import config as C
from robot_environment import RobotEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_env():
    return Monitor(RobotEnv(device=DEVICE))


def run_episode(model, env):
    """Roll out one deterministic episode and return time, position, strain, voltage."""
    obs, _ = env.reset()
    eps, sgvs, mvs = [obs[0].item()], [obs[1].item()], []
    total_reward, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        mvs.append(float(action[0]))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        eps.append(obs[0].item())
        sgvs.append(obs[1].item())
        done = terminated or truncated
    dt = env.unwrapped.time_step_delta
    return np.arange(len(eps)) * dt, np.array(eps), np.array(sgvs), np.array(mvs), total_reward


def plot_episode(t, eps, sgvs, mvs, path, show):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].step(t[:-1], mvs, where="post", color="black"); ax[0].set_ylabel("Motor voltage (V)")
    ax[1].plot(t, eps, color="blue"); ax[1].axhline(C.TARGET_POSITION, ls="--", color="green", label="target")
    ax[1].set_ylabel("Encoder position (deg)"); ax[1].legend()
    ax[2].plot(t, sgvs, color="red"); ax[2].axhline(C.STRAIN_OFFSET, ls="--", color="purple", label="rest offset")
    ax[2].set_ylabel("Strain gauge (V)"); ax[2].set_xlabel("Time (s)"); ax[2].legend()
    for a in ax:
        a.grid(True)
    fig.suptitle("SAC agent - simulated episode")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot -> {path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=C.SAC_TOTAL_TIMESTEPS)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    env = make_vec_env(make_env, n_envs=1)
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], qf=[256, 256]))
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, gamma=0.99, tau=0.005,
        buffer_size=100_000, batch_size=256,
        train_freq=(1, "step"), gradient_steps=1,
        policy_kwargs=policy_kwargs, verbose=1, seed=42, device=DEVICE,
        tensorboard_log=os.path.join(C.LOGS_DIR, "tensorboard"),
    )

    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=C.SAC_BEST_AGENT_DIR,
        log_path=os.path.join(C.LOGS_DIR, "eval_results"),
        eval_freq=1000, n_eval_episodes=5, verbose=1,
    )

    print("\n--- Training SAC agent ---")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)
    model.save(C.SAC_AGENT_PATH)
    print(f"Saved agent -> {C.SAC_AGENT_PATH}.zip")

    rewards, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True)
    print(f"Mean reward over 10 episodes: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    t, eps, sgvs, mvs, total = run_episode(model, eval_env)
    print(f"Visualisation episode reward {total:.2f} | final position {eps[-1]:.2f} deg | "
          f"final strain {sgvs[-1]:.4f} V")
    plot_episode(t, eps, sgvs, mvs, os.path.join(C.RESULTS_DIR, "sac_simulated_episode.png"), not args.no_show)

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
