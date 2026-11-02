"""Step 1 - System identification: train the LSTM model of the flexible arm.

Loads every CSV run in the data folder, builds 50-step sliding windows,
trains a 2-layer LSTM (256 units) with early stopping, saves the model and
the input/output scalers, reports the test RMSE and finally runs an
open-loop simulation for a pulse input to check the learned dynamics.

Usage:
    python src/train_lstm_model.py --data-dir data/runs
"""
import argparse
import copy
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config as C
from robot_lstm import RobotLSTM, simulate_robot_dynamics

COLUMN_NAMES = {
    "Time (s)": "Time",
    "Motor Voltage (V)": "Motor_Voltage",
    "Encoder Position (Degrees)": "Encoder_Position",
    "Strain Gauge Voltage (V)": "Strain_Gauge_Voltage",
}
INPUT_COLS = ["Motor_Voltage", "Encoder_Position", "Strain_Gauge_Voltage"]
TARGET_COLS = ["Encoder_Position", "Strain_Gauge_Voltage"]


def load_runs(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    print(f"Loading {len(files)} CSV files from '{data_dir}' ...")
    runs = []
    for i, name in enumerate(files):
        path = os.path.join(data_dir, name)
        try:
            df = pd.read_csv(path).rename(columns=COLUMN_NAMES)
            df = df[df["Time"] >= C.DATA_START_TIME_S].reset_index(drop=True)
            runs.append(df)
        except Exception as e:  # keep going if one file is broken
            print(f"  skipping {name}: {e}")
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  processed {i + 1}/{len(files)}")
    if not runs:
        raise ValueError(f"No runs could be loaded from '{data_dir}'.")
    return runs


def create_lstm_sequences(df, look_back):
    """Sliding windows inside ONE run (windows never cross two experiments)."""
    inputs = df[INPUT_COLS].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLS].to_numpy(dtype=np.float32)
    n = len(df) - look_back
    if n <= 0:
        return np.empty((0, look_back, len(INPUT_COLS))), np.empty((0, len(TARGET_COLS)))
    X = np.stack([inputs[i:i + look_back] for i in range(n)])
    y = targets[look_back:look_back + n]
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    parser.add_argument("--epochs", type=int, default=C.LSTM_EPOCHS)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = parser.parse_args()

    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- data ----------------
    runs = load_runs(args.data_dir)
    seqs = [create_lstm_sequences(df, C.LOOK_BACK_STEPS) for df in runs]
    X_sequences = np.concatenate([s[0] for s in seqs])
    y_targets = np.concatenate([s[1] for s in seqs])
    print(f"X: {X_sequences.shape}  y: {y_targets.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_targets, test_size=0.2, random_state=42)
    n_feat = X_train.shape[2]

    scaler_X = StandardScaler().fit(X_train.reshape(-1, n_feat))
    scaler_y = StandardScaler().fit(y_train)
    X_train_s = scaler_X.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test_s = scaler_X.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)
    y_train_s = scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_train_t, y_train_t, X_test_t, y_test_t = map(to_t, (X_train_s, y_train_s, X_test_s, y_test_s))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=C.LSTM_BATCH_SIZE, shuffle=True)

    # ---------------- training ----------------
    model = RobotLSTM(n_feat, C.HIDDEN_SIZE, C.OUTPUT_SIZE, num_layers=C.NUM_LSTM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=C.LSTM_LR)

    best_val_loss, epochs_no_improve, best_state = float("inf"), 0, None
    print("\n--- Training LSTM (early stopping) ---")
    for epoch in range(args.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()

        if val_loss < best_val_loss - C.LSTM_MIN_DELTA:
            best_val_loss, epochs_no_improve = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch + 1}/{args.epochs}  train {loss.item():.4f}  val {val_loss:.4f}")
        if epochs_no_improve >= C.LSTM_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), C.LSTM_MODEL_PATH)
    joblib.dump(scaler_X, C.SCALER_X_PATH)
    joblib.dump(scaler_y, C.SCALER_Y_PATH)
    print(f"Saved model -> {C.LSTM_MODEL_PATH}\nSaved scalers -> {C.SCALER_X_PATH}, {C.SCALER_Y_PATH}")

    # ---------------- evaluation ----------------
    model.eval()
    with torch.no_grad():
        y_pred = scaler_y.inverse_transform(model(X_test_t).cpu().numpy())
    rmse_enc = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_sg = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    print(f"\nTest RMSE  encoder position: {rmse_enc:.4f} deg   strain voltage: {rmse_sg:.4f} V")

    # ---------------- open-loop simulation of a pulse input ----------------
    fs = C.SAMPLING_RATE_HZ
    pattern = np.concatenate([np.full(int(0.2 * fs), 0.0),
                              np.full(int(0.4 * fs), -0.36),
                              np.full(int(3.3 * fs), 0.0)])
    init = np.zeros((C.LOOK_BACK_STEPS, 3))
    init[:, 2] = np.mean([df["Strain_Gauge_Voltage"].mean() for df in runs])
    sim = simulate_robot_dynamics(model, init, pattern, scaler_X, scaler_y, device)
    t = np.arange(len(pattern)) / fs

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].step(t, pattern, where="post", color="black"); ax[0].set_ylabel("Motor voltage (V)")
    ax[1].plot(t, sim[:, 0], color="blue"); ax[1].set_ylabel("Encoder position (deg)")
    ax[2].plot(t, sim[:, 1], color="red"); ax[2].set_ylabel("Strain gauge (V)"); ax[2].set_xlabel("Time (s)")
    for a in ax:
        a.grid(True)
    fig.suptitle("LSTM simulator - response to a pulse input")
    fig.tight_layout()
    out = os.path.join(C.RESULTS_DIR, "lstm_pulse_simulation.png")
    fig.savefig(out, dpi=120)
    print(f"Saved plot -> {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
