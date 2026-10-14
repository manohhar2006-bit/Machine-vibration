# System Run & Verification Guide
## Predictive Machine Vibration Monitoring

This guide provides step-by-step instructions for operating the system in two modes:
1. **Simulation Mode** (No hardware required; runs entirely on your laptop).
2. **Physical Hardware Mode** (Arduino Uno R3, ADXL345 accelerometer, IR tachometer, L298N driver, 555 DC motor).

---

## 1. Prerequisites and Python Setup

Ensure Python 3.11+ is installed. Open PowerShell or Command Prompt in the repository root:

```bash
cd c:\Users\manoh\Downloads\Machine-Vibration-Monitoring-main\Machine-Vibration-Monitoring-main

# 1. Install required packages
pip install -r requirements.txt
```

Verify that all dependencies and signal processing tests pass:
```bash
python tests/test_pipeline.py -v
```
*(All 17 tests should display `ok`).*

---

## 2. Mode A: Simulation Mode (Laptop Demo)

Simulation mode replays pre-calibrated vibration windows through the Flask API, exercising feature extraction, baseline standardization, the neural network classifier, and the live dashboard without needing an assembled test rig.

### Step 1: Generate Cached Demo Windows
```bash
python serving/build_demo_cache.py
```
*Output: `Successfully generated and saved demo cache -> data/processed/demo_windows.npz`.*

### Step 2: Start the Flask Backend Server
In **Terminal 1**:
```bash
python serving/app.py
```
Leave this terminal running. The server starts at `http://127.0.0.1:5000`.

### Step 3: Open the Dashboard
Open your web browser and navigate to:
**`http://127.0.0.1:5000`**

You will see:
- Connection Status: `LIVE` (green pill)
- Calibration Status: `Awaiting Calibration`
- Live Waveform and FFT spectrum canvases ready.

### Step 4: Stream Demo Vibration Windows
In **Terminal 2**:
```bash
python serving/simulate_stream.py --interval 0.5
```

### Expected Behavior on Dashboard
1. **Calibration Phase**: The backend accumulates 10 healthy normal windows and fits the adaptive baseline with feature scale floors.
2. **Unseen Normal Windows (1–15)**: The verdict displays:
   - Status: **`NORMAL OPERATION`** (Green)
   - Model Confidence: Low fault probability ($2\% - 14\%$).
3. **Imbalance Fault Windows (1–25)**: The verdict immediately flips:
   - Status: **`FAULTY -- ROTOR IMBALANCE`** (Red)
   - Model Confidence: $100.0\%$.
   - Live FFT spectrum will clearly show elevated 1X rotational vibration peak.

---

## 3. Mode B: Physical Hardware Operation

Follow this staged startup procedure to prevent unexpected motor starts and ensure measurement integrity.

### Stage 1: Firmware Compilation and Upload
1. Connect your **Arduino Uno R3** to the laptop via USB.
2. Identify the assigned COM port:
   ```bash
   python serving/serial_reader.py --list-ports
   ```
   *(Note down the port, e.g. `COM3` or `COM4`).*
3. Compile and upload the firmware:
   - **Using Arduino CLI**:
     ```bash
     "C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe" compile --fqbn arduino:avr:uno firmware/vibration_monitor
     "C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe" upload -p COM3 --fqbn arduino:avr:uno firmware/vibration_monitor
     ```
   - **Using Arduino IDE GUI**:
     Open `firmware/vibration_monitor/vibration_monitor.ino`, select Board **Arduino Uno**, select Port **COM3**, and click **Upload**.

### Stage 2: Hardware Safety & Electrical Pre-Checks
- [ ] 12V DC power supply adapter is **UNPLUGGED**.
- [ ] Motor disc is free from mechanical obstructions.
- [ ] No imbalance test weights are attached yet (pure baseline state).
- [ ] Common ground is verified between Arduino GND and L298N GND.
- [ ] ADXL345 is firmly affixed to the motor bearing assembly.

### Stage 3: Start Host Software
1. **Start the Flask Backend** (Terminal 1):
   ```bash
   python serving/app.py
   ```
2. **Open the Dashboard**:
   Navigate to `http://127.0.0.1:5000`.
3. **Start the Serial Bridge** (Terminal 2):
   ```bash
   python serving/serial_reader.py --port COM3
   ```
   *The serial bridge connects and outputs:*
   ```
   Connected to COM3 successfully.
   [Firmware Info] $BOOT,Predictive Machine Vibration Monitoring Uno R3 Initializing...
   [Firmware Info] $OK,ADXL345_INITIALIZED_800HZ
   [Firmware Info] $READY,Waiting for commands or sampling...
   ```

### Stage 4: Controlled Motor Startup
1. Plug in the 12V DC motor power adapter.
2. Confirm the motor **remains stationary** (the firmware defaults strictly to STOPPED).
3. On the dashboard, locate the **Motor Control Panel**:
   - Set PWM speed slider to `180` (~70% duty cycle).
   - Click **Start Motor**.
4. The motor begins rotating smoothly. The **Actual Shaft Speed** metric on the dashboard will update with measured RPM (e.g. `1750.0 RPM`).
5. Verify that the time-domain waveform shows physical vibration oscillations.

### Stage 5: Healthy Baseline Calibration
1. Allow the motor to stabilize at its set speed for 5–10 seconds.
2. Click **Start Calibration** on the dashboard.
3. The dashboard status updates to `Calibrating (N windows)`. Allow it to collect at least 5 to 10 windows.
4. Click **Finalize & Save Baseline**.
5. The status displays `Calibrated`. The machine operational status updates to:
   **`NORMAL OPERATION`** (Green).

### Stage 6: Controlled Imbalance Fault Injection Test
1. Click **Stop Motor** on the dashboard to safely de-energize the motor driver.
2. Wait for the shaft to come to a complete stop.
3. Attach a small test imbalance weight (e.g., M3 or M4 screw/nut) to the outer radius of the rotating disc.
4. Verify the safety guard is in place over the disc.
5. Click **Start Motor** on the dashboard.
6. Observe the dashboard:
   - Vibration RMS increases noticeably.
   - FFT spectrum shows a pronounced 1X rotational harmonic peak.
   - Status flips to **`FAULTY -- ROTOR IMBALANCE`** (Red).
   - Model Confidence displays high fault certainty.
7. Click **Stop Motor** when testing is concluded.

---

## 4. Troubleshooting Reference

| Symptom | Probable Cause | Remediation |
|---|---|---|
| Dashboard says `OFFLINE` (Red dot) | Flask backend not running | Start `python serving/app.py` in a terminal. |
| Dashboard says `STALE` (Amber dot) | Serial reader stopped or Arduino disconnected | Verify USB cable; ensure `serial_reader.py` is running and transmitting packets. |
| `$ERR,ADXL345_NOT_FOUND` in serial logs | I2C wiring loose or address pin miswired | Check SDA &rarr; A4, SCL &rarr; A5, SDO &rarr; GND (for 0x53). Ensure ADXL345 VCC is powered. |
| RPM reads `0.0` while motor rotates | IR sensor misalignment or bad interrupt pin | Ensure sensor OUT connects to Pin 2. Check optical disc passes cleanly through slot. |
| Motor does not rotate when clicking Start | L298N ENA jumper still installed, or 12V unpowered | Remove the ENA jumper cap from L298N. Verify 12V adapter is energized and common GND is wired. |
| Motor rotates at 100% duty cycle continuously | ENA jumper still bridging ENA to 5V | Remove the black shunt jumper cap on L298N `ENA` pin so Arduino Pin 9 controls PWM. |
| Calibration fails with `Insufficient windows` | Clicked stop immediately after start | Allow at least 3 to 5 seconds between clicking Start and Stop Calibration. |
