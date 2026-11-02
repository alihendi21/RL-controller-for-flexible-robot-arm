"""Shared configuration for the whole pipeline.

Every script imports its constants and file paths from here, so the LSTM
training, the Gym environment, the SAC training and the export step always
agree on file names and hyper-parameters.
"""
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder with the experimental CSV logs (one file per run, 100 Hz).
# Expected columns: "Time (s)", "Motor Voltage (V)",
#                   "Encoder Position (Degrees)", "Strain Gauge Voltage (V)"
DATA_DIR = os.path.join(ROOT_DIR, "data", "runs")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "robot_lstm_model.pth")
SCALER_X_PATH = os.path.join(MODELS_DIR, "scaler_x.joblib")
SCALER_Y_PATH = os.path.join(MODELS_DIR, "scaler_y.joblib")
SAC_AGENT_PATH = os.path.join(MODELS_DIR, "final_robot_sac_agent")      # .zip is added by SB3
SAC_BEST_AGENT_DIR = os.path.join(LOGS_DIR, "best_rl_model")              # best_model.zip (EvalCallback)

# File that is copied to the Raspberry Pi and played back on the real arm.
ACTION_SEQUENCE_CSV = os.path.join(RESULTS_DIR, "generated_action_sequence.csv")

# --------------------------------------------------------------------------
# Data / LSTM system identification
# --------------------------------------------------------------------------
SAMPLING_RATE_HZ = 100
DATA_START_TIME_S = 2.0          # first 2 s of every run (idle) are discarded
LOOK_BACK_STEPS = 50             # 0.5 s of history at 100 Hz

INPUT_SIZE = 3                   # [motor voltage, encoder position, strain voltage]
OUTPUT_SIZE = 2                  # [encoder position, strain voltage] at t+1
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2

LSTM_EPOCHS = 100
LSTM_PATIENCE = 20
LSTM_MIN_DELTA = 2e-4
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3

# --------------------------------------------------------------------------
# Reinforcement learning task
# --------------------------------------------------------------------------
TARGET_POSITION = -30.0          # degrees
STRAIN_OFFSET = 2.4              # strain-gauge voltage of the link at rest (V)
MAX_EPISODE_STEPS = 400          # 4 s at 100 Hz
MOTOR_VOLTAGE_MIN = -1.25
MOTOR_VOLTAGE_MAX = 1.25

SAC_TOTAL_TIMESTEPS = 600_000
