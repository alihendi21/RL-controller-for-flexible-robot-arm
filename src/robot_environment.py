"""Step 2 - Gymnasium environment built on top of the trained LSTM simulator.

Observation (4,): [encoder position, strain voltage,
                   position error to target, strain deviation from rest offset]
Action      (1,): motor voltage in [-1.25, 1.25] V
Goal: move the link to TARGET_POSITION (-30 deg) and damp the tip vibration
(strain voltage back to STRAIN_OFFSET) with as little voltage as possible.
"""
import os

import gymnasium as gym
import joblib
import numpy as np
import torch
from gymnasium import spaces

import config as C
from robot_lstm import RobotLSTM, simulation_step


class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": C.SAMPLING_RATE_HZ}

    def __init__(self,
                 trained_model_path=C.LSTM_MODEL_PATH,
                 scaler_x_path=C.SCALER_X_PATH,
                 scaler_y_path=C.SCALER_Y_PATH,
                 look_back=C.LOOK_BACK_STEPS,
                 device="cpu",
                 sampling_rate_hz=C.SAMPLING_RATE_HZ,
                 target_position=C.TARGET_POSITION,
                 strain_offset=C.STRAIN_OFFSET,
                 max_episode_steps=C.MAX_EPISODE_STEPS,
                 initial_pos_range=(-3, 3),
                 initial_strain_offset_range=(-0.03, 0.03)):
        super().__init__()
        for path in (trained_model_path, scaler_x_path, scaler_y_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found - run train_lstm_model.py first.")

        self.look_back = look_back
        self.device = torch.device(device)
        self.time_step_delta = 1.0 / sampling_rate_hz

        self.model = RobotLSTM(C.INPUT_SIZE, C.HIDDEN_SIZE, C.OUTPUT_SIZE, num_layers=C.NUM_LSTM_LAYERS).to(self.device)
        self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device))
        self.model.eval()
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

        self.target_position = target_position
        self.strain_offset = strain_offset
        self.max_episode_steps = max_episode_steps
        self.initial_pos_range = initial_pos_range
        self.initial_strain_offset_range = initial_strain_offset_range

        obs_low = np.array([-360.0, 0.0, -390.0, -2.5], dtype=np.float32)
        obs_high = np.array([360.0, 5.0, 390.0, 2.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(4,), dtype=np.float32)

        self.min_motor_voltage = C.MOTOR_VOLTAGE_MIN
        self.max_motor_voltage = C.MOTOR_VOLTAGE_MAX
        self.action_space = spaces.Box(low=self.min_motor_voltage, high=self.max_motor_voltage,
                                       shape=(1,), dtype=np.float32)

        self.current_step = 0
        self.current_encoder_pos = 0.0
        self.current_strain_gauge_voltage = 0.0
        self.current_motor_voltage = 0.0
        self.current_sequence_raw = np.zeros((self.look_back, C.INPUT_SIZE))

    def _get_observation(self):
        return np.array([
            self.current_encoder_pos,
            self.current_strain_gauge_voltage,
            self.target_position - self.current_encoder_pos,
            self.current_strain_gauge_voltage - self.strain_offset,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_encoder_pos = self.np_random.uniform(*self.initial_pos_range)
        self.current_strain_gauge_voltage = self.strain_offset + self.np_random.uniform(*self.initial_strain_offset_range)
        self.current_motor_voltage = 0.0
        self.current_sequence_raw = np.stack([
            np.full(self.look_back, self.current_motor_voltage),
            np.full(self.look_back, self.current_encoder_pos),
            np.full(self.look_back, self.current_strain_gauge_voltage),
        ], axis=1)
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        motor_voltage = float(np.clip(action, self.min_motor_voltage, self.max_motor_voltage).item())

        predicted, self.current_sequence_raw = simulation_step(
            self.model, self.current_sequence_raw, motor_voltage,
            self.scaler_X, self.scaler_y, self.device)
        self.current_encoder_pos = float(predicted[0])
        self.current_strain_gauge_voltage = float(predicted[1])
        self.current_motor_voltage = motor_voltage

        # ---------------- reward ----------------
        pos_error = abs(self.target_position - self.current_encoder_pos)
        pos_tolerance = 5.0
        reward_pos = -2.0 * pos_error ** 2
        if pos_error < pos_tolerance:
            reward_pos += 30.0

        sgv_deviation_abs = abs(self.current_strain_gauge_voltage - self.strain_offset)
        reward_strain = -500.0 * sgv_deviation_abs
        reward_time = -0.01
        reward_action = -60.0 * motor_voltage ** 2
        reward = reward_pos + reward_strain + reward_time + reward_action

        strain_damped_tolerance = 0.1
        if pos_error < pos_tolerance and sgv_deviation_abs < strain_damped_tolerance:
            # bonus for holding the target with a calm link and ~0 V
            reward += 250.0 + 320.0 / (1.0 + 50.0 * abs(motor_voltage)) + 1.0

        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        if truncated and (pos_error > pos_tolerance or sgv_deviation_abs > strain_damped_tolerance):
            reward -= 200.0

        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
