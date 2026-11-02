# Data

`data/runs/` contains the **520 experimental runs** used to train the LSTM model. They are one CSV per run, logged on the Raspberry Pi at about 100 Hz:

| Time (s) | Motor Voltage (V) | Encoder Position (Degrees) | Strain Gauge Voltage (V) |
|---|---|---|---|
| 0.0006 | 0.0 | 0.0 | 0.0 |
| 0.0111 | 0.0 | 0.0 | 2.3693 |
| … | … | … | … |

- Each run lasts about 6 s: 2 s at rest, then the excitation (pulse, step or sine), then free vibration.
- The first 2 s are discarded during training (`DATA_START_TIME_S` in `src/config.py`). The first sample of each run has a strain value of 0 because the ADC isn't ready yet.
- The motor voltage is the command sent to the drive through the DAC (±1.25 V range). The encoder position is in degrees from the start position, and the strain-gauge voltage is the bridge output from the ADS1115 (about 2.4 V at rest).
- New runs recorded with `raspberry_pi/run_on_robot.py` or `raspberry_pi/collect_pulse_runs.py` use the same format.
