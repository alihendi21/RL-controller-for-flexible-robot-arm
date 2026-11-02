"""Play a motor-voltage sequence on the real flexible arm and log the sensors.

Runs on a Raspberry Pi 4B:
  * MCP4725 DAC (I2C 0x60)  -> analog command to the servo drive (0-5 V, 2.5 V = 0 motor volts)
  * ADS1115 ADC (I2C 0x49)  <- strain-gauge bridge, differential A0-A1
  * Incremental encoder     <- GPIO17 (A) / GPIO27 (B)

Threads: DAC command playback, encoder logger, strain-gauge sampler,
CSV logger (100 Hz) and an optional live plot.

The same script is used for
  1. data collection  - play an excitation profile from src/generate_excitation.py
  2. deployment       - play results/generated_action_sequence.csv from the SAC agent

Usage (on the Pi):
    python3 run_on_robot.py --input generated_action_sequence.csv
    python3 run_on_robot.py --input step_000.csv --output-prefix run --no-plot
"""
import argparse
import csv
import threading
import time
from datetime import datetime

import adafruit_ads1x15.ads1115 as ADS
import board
import busio
import matplotlib.pyplot as plt
import numpy as np
import RPi.GPIO as GPIO
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_mcp4725 import MCP4725
from matplotlib.animation import FuncAnimation

MOTOR_VOLTAGE_MIN = -1.25
MOTOR_VOLTAGE_MAX = 1.25
PULSES_PER_REVOLUTION = 33000
ENCODER_PIN_A = 17
ENCODER_PIN_B = 27
LOG_INTERVAL_S = 0.01            # 100 Hz
STRAIN_SAMPLE_INTERVAL_S = 0.005
SETTLE_AFTER_SEQUENCE_S = 2.0    # keep logging after the last command (--no-plot mode)


def motor_voltage_to_dac_normalized(motor_voltage):
    """Map [-1.25, 1.25] motor volts to the DAC's normalized [0, 1] range."""
    clipped = np.clip(motor_voltage, MOTOR_VOLTAGE_MIN, MOTOR_VOLTAGE_MAX)
    return (clipped - MOTOR_VOLTAGE_MIN) / (MOTOR_VOLTAGE_MAX - MOTOR_VOLTAGE_MIN)


# ------------------------------------------------------------------ state
dac = None
ads = None
csv_file = None
csv_writer = None
start_time = 0.0
last_motor_voltage = 0.0
encoder_position = 0
last_a_state = 0
last_b_state = 0
stop_event = threading.Event()

time_data_encoder, degree_data = [], []
time_data_strain, strain_data = [], []
motor_voltages_to_apply = []
sequence_duration = 0.0
time_step_between_commands = 0.01


def initialize_hardware(output_prefix):
    global dac, ads, csv_file, csv_writer, start_time
    print("Initializing hardware...")
    i2c = busio.I2C(board.SCL, board.SDA)
    dac = MCP4725(i2c, address=0x60)
    dac.normalized_value = 0.5                     # 0 V to the motor

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ENCODER_PIN_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(ENCODER_PIN_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(ENCODER_PIN_A, GPIO.BOTH, callback=encoder_callback)
    GPIO.add_event_detect(ENCODER_PIN_B, GPIO.BOTH, callback=encoder_callback)

    ads = ADS.ADS1115(i2c, address=0x49)
    ads.gain = 1

    filename = f"{output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # Same header as the training data -> logs can go straight into data/runs/
    csv_writer.writerow(["Time (s)", "Motor Voltage (V)", "Encoder Position (Degrees)", "Strain Gauge Voltage (V)"])
    print(f"Logging to {filename}")
    start_time = time.time()


def cleanup_hardware():
    print("\nCleaning up hardware...")
    stop_event.set()
    if dac:
        dac.normalized_value = 0.5
    time.sleep(0.05)
    if csv_file:
        csv_file.close()
    GPIO.cleanup()
    print("Done - motor at 0 V, log file closed.")


def encoder_callback(channel):
    global encoder_position, last_a_state, last_b_state
    a_state = GPIO.input(ENCODER_PIN_A)
    b_state = GPIO.input(ENCODER_PIN_B)
    if a_state != last_a_state:
        encoder_position += -1 if a_state == b_state else 1
    last_a_state, last_b_state = a_state, b_state


def _trim(times, values):
    max_points = 2 * int(sequence_duration / time_step_between_commands) if time_step_between_commands > 0 else 1000
    max_points = max(max_points, 1000)
    while len(times) > max_points:
        times.pop(0)
        values.pop(0)


def strain_gauge_data():
    chan = AnalogIn(ads, ADS.P0, ADS.P1)
    while not stop_event.is_set():
        try:
            time_data_strain.append(time.time() - start_time)
            strain_data.append(chan.voltage)
            _trim(time_data_strain, strain_data)
            time.sleep(STRAIN_SAMPLE_INTERVAL_S)
        except Exception as e:
            print(f"Strain-gauge thread error: {e}")
            break


def encoder_data_logger():
    while not stop_event.is_set():
        time_data_encoder.append(time.time() - start_time)
        degree_data.append(encoder_position / PULSES_PER_REVOLUTION * 360.0)
        _trim(time_data_encoder, degree_data)
        time.sleep(LOG_INTERVAL_S)


def load_motor_voltages_from_csv(filepath):
    global sequence_duration, time_step_between_commands
    times, voltages = [], []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        t_idx, mv_idx = header.index("Time (s)"), header.index("Motor Voltage (V)")
        for row in reader:
            times.append(float(row[t_idx]))
            voltages.append(float(row[mv_idx]))
    if len(times) > 1:
        time_step_between_commands = times[1] - times[0]
    sequence_duration = (times[-1] + time_step_between_commands) if times else 0.0
    print(f"Loaded {len(voltages)} commands from {filepath} "
          f"(dt = {time_step_between_commands:.4f} s, duration = {sequence_duration:.2f} s)")
    return voltages


def control_voltage_from_csv():
    """Send the commands to the DAC with a fixed period (drift-compensated)."""
    global last_motor_voltage
    print("Starting motor command sequence...")
    t0 = time.perf_counter()
    for i, mv in enumerate(motor_voltages_to_apply):
        if stop_event.is_set():
            break
        dac.normalized_value = float(motor_voltage_to_dac_normalized(mv))
        last_motor_voltage = mv
        if i % 50 == 0:
            print(f"step {i}/{len(motor_voltages_to_apply) - 1}  motor voltage {mv:+.4f} V")
        next_tick = t0 + (i + 1) * time_step_between_commands
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    dac.normalized_value = 0.5
    last_motor_voltage = 0.0
    print("Sequence finished - motor set to 0 V.")


def csv_logger():
    while not stop_event.is_set():
        try:
            degrees = encoder_position / PULSES_PER_REVOLUTION * 360.0
            strain_val = strain_data[-1] if strain_data else 0.0
            csv_writer.writerow([round(time.time() - start_time, 4), round(last_motor_voltage, 3),
                                 round(degrees, 2), round(strain_val, 4)])
            time.sleep(LOG_INTERVAL_S)
        except Exception as e:
            print(f"CSV logger error: {e}")
            break


def main():
    global motor_voltages_to_apply
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generated_action_sequence.csv")
    parser.add_argument("--output-prefix", default="robot_output_from_rl_mv")
    parser.add_argument("--no-plot", action="store_true", help="headless: stop automatically after the run")
    args = parser.parse_args()

    motor_voltages_to_apply = load_motor_voltages_from_csv(args.input)
    if not motor_voltages_to_apply:
        print("No motor voltages loaded - exiting.")
        return

    initialize_hardware(args.output_prefix)
    try:
        for target in (strain_gauge_data, encoder_data_logger, csv_logger):
            threading.Thread(target=target, daemon=True).start()
        control_thread = threading.Thread(target=control_voltage_from_csv, daemon=True)
        control_thread.start()

        if args.no_plot:
            control_thread.join()
            time.sleep(SETTLE_AFTER_SEQUENCE_S)
            return

        for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid"):
            if style in plt.style.available:
                plt.style.use(style)
                break
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))
        line_encoder, = ax[0].plot([], [], label="Encoder Position (Degrees)", color="blue")
        ax[0].set(xlabel="Time (s)", ylabel="Position (Degrees)", title="Encoder Position Over Time",
                  xlim=(0, sequence_duration + 1.0), ylim=(-40, 40))
        ax[0].legend()
        line_strain, = ax[1].plot([], [], label="Strain Gauge Voltage (V)", color="red")
        ax[1].set(xlabel="Time (s)", ylabel="Voltage (V)", title="Strain Gauge Voltage Over Time",
                  xlim=(0, sequence_duration + 1.0), ylim=(1, 3))
        ax[1].legend()
        plt.tight_layout()

        def update_plot(_frame):
            n_enc = min(len(time_data_encoder), len(degree_data))
            n_sg = min(len(time_data_strain), len(strain_data))
            line_encoder.set_data(time_data_encoder[:n_enc], degree_data[:n_enc])
            line_strain.set_data(time_data_strain[:n_sg], strain_data[:n_sg])
            return line_encoder, line_strain

        _ani = FuncAnimation(fig, update_plot, interval=100, blit=True, cache_frame_data=False)
        plt.show()                                   # close the window to stop logging
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cleanup_hardware()


if __name__ == "__main__":
    main()
