# Predictive Machine Vibration Monitoring

Real-time predictive vibration monitoring and fault detection for rotating machinery (brushed DC motor) using an Arduino Uno R3, ADXL345 digital accelerometer (I2C), IR optical tachometer, adaptive baseline calibration, and a scikit-learn neural network (MLP) classifier.

---

## 1. System Architecture

- **Microcontroller**: Arduino Uno R3 (ATmega328P @ 16 MHz, 2 KB SRAM).
- **Vibration Sensor**: ADXL345 digital 3-axis accelerometer on I2C (400 kHz Fast Mode, 800 Hz ODR, full resolution mode &plusmn;2g).
- **Speed Sensor**: IR slotted optical speed sensor on Pin 2 (External Interrupt INT0) with atomic pulse sampling.
- **Motor & Controller**: 555 brushed DC motor driven by an L298N dual H-bridge module with PWM speed control (Pin 9) and safe boot default (strictly STOPPED).
- **Host Communication**: USB Serial @ 115,200 baud streaming structured packets (`$VIB,<seq>,<rate>,<rpm>,<count>,<samples...>`).
- **Signal Processing**: NumPy & SciPy real FFT with Hann windowing, RMS, dominant frequency (1X peak excluding DC), sub-band spectral energy, and Fisher excess kurtosis.
- **Adaptive Calibration**: Per-deployment healthy baseline standardization with mathematical feature scale floors (`max(std, floor)`) preventing zero-variance numerical blowup.
- **Diagnostics & Dashboard**: Flask REST backend with Chart.js live waveform, FFT spectrum, real-time RPM, feature telemetry, motor controls, and stale data detection.

---

## 2. Repository Layout

```
Machine-Vibration-Monitoring/
├── firmware/
│   └── vibration_monitor/
│       └── vibration_monitor.ino         # Arduino Uno R3 acquisition & motor control sketch
├── src/
│   ├── features.py                       # Validated RMS, FFT peak, spectral energy, kurtosis
│   ├── calibration.py                    # Robust BaselineCalibrator with scale floors
│   ├── dataset.py                        # MaFaulDa loader and 800 Hz vibration synthesizer
│   ├── model.py                          # MLPClassifier factory and serialization
│   ├── train.py                          # Offline training & held-out evaluation
│   └── ablation.py                       # Fixed vs. adaptive calibration benchmark
├── serving/
│   ├── app.py                            # Flask backend API & motor command queue
│   ├── build_demo_cache.py               # Generates cached demo windows
│   ├── simulate_stream.py                # Replays calibrated demo stream into API
│   ├── serial_reader.py                  # USB Serial bridge forwarding Arduino packets
│   └── model_store/
│       ├── classifier.joblib             # Trained MLP classifier
│       └── baseline.json                 # Healthy reference baseline statistics
├── dashboard/
│   ├── templates/index.html              # Modern responsive diagnostic UI
│   └── static/dashboard.js               # Real-time Chart.js polling & motor handlers
├── tests/
│   └── test_pipeline.py                  # 17 automated tests (features, calibration, API, serial)
├── HARDWARE_WIRING_GUIDE.md              # Detailed wiring, pin mapping, power, & ground guide
├── RUN_AND_TEST_GUIDE.md                 # Complete startup and testing workflow
├── PROJECT_FIX_REPORT.md                 # Comprehensive engineering fix & audit report
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## 3. Installation & Setup

Ensure Python 3.11+ is installed.

```bash
# Install core dependencies
pip install -r requirements.txt
```

---

## 4. Quickstart: End-to-End Simulation Demo (No Hardware Required)

You can verify the entire pipeline on your laptop without physical hardware:

### Step 1: Build the Demo Cache
```bash
python serving/build_demo_cache.py
```

### Step 2: Start the Flask Backend
```bash
python serving/app.py
```
Leave this terminal running. The server listens at `http://127.0.0.1:5000`.

### Step 3: Open the Dashboard
Open `http://127.0.0.1:5000` in your web browser.

### Step 4: Stream Demo Vibration Windows
In a second terminal:
```bash
python serving/simulate_stream.py
```
Watch the dashboard:
1. It automatically calibrates on healthy baseline windows.
2. It evaluates unseen healthy normal windows (`NORMAL OPERATION`).
3. It streams unbalance fault windows (`FAULTY -- ROTOR IMBALANCE`), demonstrating live fault detection.

---

## 5. Hardware Deployment (Physical Rig)

### Step 1: Wire the Components
Refer to [HARDWARE_WIRING_GUIDE.md](file:///c:/Users/manoh/Downloads/Machine-Vibration-Monitoring-main/Machine-Vibration-Monitoring-main/HARDWARE_WIRING_GUIDE.md) for pin connections, power supplies, and common ground requirements.

### Step 2: Compile & Upload Firmware
Open `firmware/vibration_monitor/vibration_monitor.ino` in Arduino IDE or compile via `arduino-cli`:
```bash
arduino-cli compile --fqbn arduino:avr:uno firmware/vibration_monitor
arduino-cli upload -p COM3 --fqbn arduino:avr:uno firmware/vibration_monitor
```

### Step 3: Start Flask Backend & Serial Bridge
```bash
# Terminal 1: Backend
python serving/app.py

# Terminal 2: Serial Bridge
python serving/serial_reader.py --port COM3
```

---

## 6. Running Automated Tests

To run the complete automated test suite (17 tests covering signal processing, calibration numerical stability, API endpoints, and serial framing):

```bash
python tests/test_pipeline.py
```