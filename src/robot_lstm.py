"""LSTM model of the flexible arm and helpers to roll it forward in time.

The network receives the last ``LOOK_BACK_STEPS`` samples of
[motor voltage, encoder position, strain-gauge voltage] and predicts the
encoder position and strain-gauge voltage of the next sample (10 ms later).
Feeding the prediction back into the input window turns it into a
data-driven simulator of the real arm.
"""
import numpy as np
import torch
import torch.nn as nn


class RobotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc_out = nn.Linear(128, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep_output = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_timestep_output))
        return self.fc_out(x)


def predict_next_state(model, sequence_raw, scaler_X, scaler_y, device):
    """One-step prediction. ``sequence_raw`` has shape (look_back, 3), unscaled."""
    look_back, n_features = sequence_raw.shape
    x_scaled = scaler_X.transform(sequence_raw.reshape(-1, n_features)).reshape(1, look_back, n_features)
    with torch.no_grad():
        y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    return scaler_y.inverse_transform(y_scaled.reshape(1, -1))[0]


def simulation_step(model, sequence_raw, motor_voltage, scaler_X, scaler_y, device):
    """Advance the simulator by one sample.

    Returns the predicted [encoder position, strain voltage] and the updated
    input window (oldest sample dropped, newest appended).
    """
    predicted = predict_next_state(model, sequence_raw, scaler_X, scaler_y, device)
    new_row = np.array([motor_voltage, predicted[0].item(), predicted[1].item()])
    updated_sequence = np.vstack((sequence_raw[1:], new_row))
    return predicted, updated_sequence


def simulate_robot_dynamics(model, initial_sequence_raw, motor_voltage_pattern, scaler_X, scaler_y, device):
    """Open-loop simulation for a whole motor-voltage pattern."""
    model.eval()
    sequence = initial_sequence_raw.copy()
    outputs = []
    for mv in motor_voltage_pattern:
        predicted, sequence = simulation_step(model, sequence, mv, scaler_X, scaler_y, device)
        outputs.append(predicted)
    return np.array(outputs)
