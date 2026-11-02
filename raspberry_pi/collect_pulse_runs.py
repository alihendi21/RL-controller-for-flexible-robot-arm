"""Original data-collection script used on the Raspberry Pi (pulse experiments).

Each run lasts 6 s at 100 Hz: 2 s at rest, then a 0.4 s pulse on the DAC
(normalized 0.85 -> +0.875 V or 0.15 -> -0.875 V motor command), then free
vibration. Every run is saved to runs/<name>.csv with the same columns as the
training data in data/runs/. Between runs, the link is put back by hand.

This is the script used for the pulse experiments in data/runs/ (the pulse
level and width in generate_input_signal() were changed between batches).
For new step / pulse / sine experiments you can also use
src/generate_excitation.py together with run_on_robot.py.
"""
import board
import busio
import RPi.GPIO as GPIO
from adafruit_mcp4725 import MCP4725
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import threading
import time
import csv
from datetime import datetime
import sys
import os

# -----------------------------
# Hardware & Sensor Setup
# -----------------------------
i2c = busio.I2C(board.SCL, board.SDA)
dac = MCP4725(i2c, address=0x60)
dac.normalized_value = 0.5
last_motor_voltage = 0.0

ENCODER_A_PIN = 17
ENCODER_B_PIN = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(ENCODER_A_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENCODER_B_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

PULSES_PER_REVOLUTION = 33000
encoder_position = 0
last_a_state = 0
last_b_state = 0

# Encoder callback
def encoder_callback(channel):
    global encoder_position, last_a_state, last_b_state
    a = GPIO.input(ENCODER_A_PIN)
    b = GPIO.input(ENCODER_B_PIN)
    if a != last_a_state:
        if a == b:
            encoder_position -= 1
        else:
            encoder_position += 1
    last_a_state = a
    last_b_state = b

GPIO.add_event_detect(ENCODER_A_PIN, GPIO.BOTH, callback=encoder_callback)
GPIO.add_event_detect(ENCODER_B_PIN, GPIO.BOTH, callback=encoder_callback)

# Strain gauge setup
i2c_ads = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c_ads, address=0x49)
ads.gain = 1
strain_val = 0.0

# -----------------------------
# Input Voltage Pattern
# -----------------------------
def generate_input_signal(run_index, t):
    if 2.0 <= t <= 2.4:
        if run_index == 1:
            return 0.85 if t < 2.4 else 0.50
        elif run_index == 2:
            return 0.15 if t < 2.4 else 0.50
    return 0.5

# -----------------------------
# One Full Run of Robot (6s)
# -----------------------------
def run_robot(run_index):
    global encoder_position, last_motor_voltage, strain_val

    encoder_position = 0  # reset encoder count
    duration = 6.0
    sample_rate = 100 # 100Hz = 0.01s

    # Prepare CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run35_3_{run_index}.csv"
    os.makedirs("runs", exist_ok=True)
    filepath = os.path.join("runs", filename)
    csv_file = open(filepath, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Time (s)", "Motor Voltage (V)", "Encoder Position (Degrees)", "Strain Gauge Voltage (V)"])

    # Thread to read strain continuously
    def strain_reader():
        global strain_val
        while time.time() - start_time < duration:
            chan = AnalogIn(ads, ADS.P0, ADS.P1)
            strain_val = chan.voltage
            time.sleep(0.001)

    # Start
    start_time = time.time()
    strain_thread = threading.Thread(target=strain_reader)
    strain_thread.start()

    try:
        while time.time() - start_time < duration:
            t = time.time() - start_time
            # Generate input voltage
            v = generate_input_signal(run_index, t)
            dac.normalized_value = v
            last_motor_voltage = (v - 0.5) * 2.5

            degrees = (encoder_position / PULSES_PER_REVOLUTION) * 360

            writer.writerow([
                round(t, 4),
                round(last_motor_voltage, 3),
                round(degrees, 2),
                round(strain_val, 4)
            ])

            time.sleep(1 / sample_rate)

    finally:
        dac.normalized_value = 0.5
        strain_thread.join()
        csv_file.close()
        print(f"[OK] Run {run_index} saved to {filepath}")

# -----------------------------
# Main Program Loop
# -----------------------------
try:
    for run in range(1, 3):
        print(f"[INFO] Starting run {run}...")
        run_robot(run)
        print(f"[INFO] Run {run} completed. Please reset link position.")
        time.sleep(4)

    print("[OK] All runs completed.")

finally:
    GPIO.cleanup()
    sys.exit(0)

