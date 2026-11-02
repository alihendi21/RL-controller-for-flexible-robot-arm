# Trained models

These are the models used for the final experiment on the real arm:

| File | What it is | Produced by |
|---|---|---|
| `robot_lstm_model.pth` | LSTM dynamics model (PyTorch `state_dict`, 2 × 256 LSTM → 128 → 2) | `src/train_lstm_model.py` |
| `scaler_x.joblib`, `scaler_y.joblib` | scikit-learn `StandardScaler`s for the LSTM inputs and outputs | `src/train_lstm_model.py` |
| `final_robot_sac_agent.zip` | Stable-Baselines3 SAC agent (600k steps) | `src/train_sac_agent.py` |

`experiments/generated_action_sequence.csv` was generated from this agent and played on the robot.

Notes:
- Running the training scripts again overwrites these files. Make a copy first if you want to keep them.
- The scalers were saved with an older scikit-learn version. Loading them with a newer one prints an `InconsistentVersionWarning`, which is harmless here.
