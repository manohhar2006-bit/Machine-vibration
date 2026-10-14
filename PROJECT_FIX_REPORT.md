# Project Fix Report: Predictive Machine Vibration Monitoring
**Embedded Systems Review 1 Implementation & Defect Remediation**

**Date:** October 14, 2026  
**Target Platform:** Arduino Uno R3 (ATmega328P), ADXL345 Accelerometer, IR Slotted Optical Speed Sensor, L298N Driver, 555 DC Motor, Flask Backend, Chart.js Dashboard.

---

## 1. Executive Summary

This report documents the comprehensive defect remediation, architecture alignment, hardware firmware redesign, signal processing overhaul, and end-to-end integration performed on the **Predictive Machine Vibration Monitoring** project.

Prior to these fixes, the project suffered from a fatal firmware memory defect that caused immediate compilation failure on the target Arduino Uno (requesting 16,000 bytes on a 2,048-byte SRAM device), non-existent motor control integration, unsafe interrupt race conditions, catastrophic numerical instability in the adaptive baseline calibration (producing $10^8$ multipliers on fractional frequency jitter), unhandled `NaN` crashes on sensor flatline, redundant signal processing calculations, and an absence of automated testing.

All critical defects have been resolved directly within the repository codebase. The firmware now compiles cleanly for the Arduino Uno using only 50% of available dynamic SRAM (1,029 bytes), the adaptive baseline calibrator incorporates mathematically and physically justified scale floors, feature extraction safely handles edge cases without `NaN`, the Flask API and serial bridge implement bidirectional motor control and stale data detection, and a suite of 17 automated tests verifies the pipeline.

---

## 2. Problems Identified and Root Causes

| Defect ID | Component | Defect Description | Root Cause |
|---|---|---|---|
| **DEF-01** | `firmware/vibration_monitor.ino` | Firmware failed to compile with `Error: data section exceeds available space in board` (16,434 bytes requested). | Code allocated `float accelBuffer[4000];` (16 KB) on an ATmega328P microcontroller with only 2 KB (2,048 bytes) total SRAM. |
| **DEF-02** | `firmware/vibration_monitor.ino` | Unrealistic sampling rate (4,000 Hz) polled against factory default 100 Hz sensor ODR over standard 100 kHz I2C. | The ADXL345 `BW_RATE` register was left unconfigured at 100 Hz. Polling at 4,000 Hz produced 40 identical repeated reads per actual measurement. |
| **DEF-03** | `firmware/vibration_monitor.ino` | Motor driver completely unintegrated; L298N pins unassigned; no start/stop safety. | Firmware had no pin definitions or control logic for L298N ENA, IN1, IN2. Hardware was at risk of unintended energization. |
| **DEF-04** | `firmware/vibration_monitor.ino` | Non-atomic reading of optical tachometer pulse counts in `loop()`. | `pulseCount` is a 4-byte `unsigned long` on an 8-bit AVR architecture. Reads without `noInterrupts()` caused race conditions and corrupted RPM values. |
| **DEF-05** | `src/calibration.py` | Division-by-epsilon causing extreme numerical instability ($10^8$ z-scores) and false alarms. | Dominant frequency standard deviation under steady-state speed is 0.0. Calibrator divided $(x - \text{mean}) / (0 + 10^{-8})$, blowing up normal variations. |
| **DEF-06** | `src/features.py` | Sensor flatline (zeros or constant DC) crashed the server with `ValueError: Input X contains NaN`. | Fisher kurtosis on constant arrays suffers division by zero in `scipy.stats.kurtosis`. Absence of input validation permitted NaNs to reach the MLP classifier. |
| **DEF-07** | `src/features.py` & `serving/app.py` | Severe spectral leakage in FFT; redundant double FFT calculation per ingested window. | Time windows were transformed with a rectangular window without mean detrending or Hann windowing. FFT was computed in `extract_features` and recomputed in `app.py`. |
| **DEF-08** | `serving/serial_reader.py` | Fragile parsing, no connection recovery, no port auto-discovery, no command transmission. | Serial reader assumed comma-separated ASCII without framing headers, lacked exception handling for port disconnection, and could not forward motor commands. |
| **DEF-09** | `serving/build_demo_cache.py` | Crash with `FileNotFoundError` when raw 10 GB Kaggle MaFaulDa files were absent. | Cache builder had no fallback generator for users without local copies of the raw dataset. |
| **DEF-10** | `dashboard/` | Minimal dashboard without motor controls, stale connection warnings, or confidence differentiation. | Interface did not differentiate model confidence from machine health, did not show disconnected states, and lacked motor start/stop triggers. |

---

## 3. Files Modified and Summary of Fixes

### 3.1. `firmware/vibration_monitor.ino` and `firmware/vibration_monitor/vibration_monitor.ino`
- **Memory Redesign**: Replaced the 16,000-byte buffer with `int16_t accelBuffer[256];` storing raw 16-bit accelerometer counts. The entire sketch now consumes **1,029 bytes of SRAM (50%)** and **8,420 bytes of Flash (26%)**, leaving 1,019 bytes free for local variables and the execution stack.
- **Sensor Synchronization**: Configured ADXL345 register `0x2C` (`BW_RATE`) to `0x0D` (800 Hz ODR, 400 Hz analog bandwidth), register `0x31` (`DATA_FORMAT`) to `0x08` (full resolution, $\pm 2$g range), and register `0x2D` (`POWER_CTL`) to `0x08` (measurement mode). Configured I2C clock to 400 kHz Fast Mode. Verified device identification register `0x00` against expected ID `0xE5`.
- **Safe Motor Control**: Wired L298N pins (Pin 9 for Timer 1 PWM speed enable `ENA`, Pin 8 for `IN1`, Pin 7 for `IN2`). Initialized all pins as `OUTPUT` and `LOW` during `setup()`. Implemented non-blocking serial command parser supporting `MOTOR START [speed]`, `MOTOR STOP`, `MOTOR SPEED <0-255>`, and `STATUS`.
- **Tachometer Race Protection**: Protected `pulseCount` in `updateRpm()` with `noInterrupts()` and `interrupts()`, using configurable pulses-per-revolution (`PULSES_PER_REV = 20`).
- **Framed Protocol**: Replaced raw CSV lines with structured packet framing:
  `$VIB,<seq>,<sample_rate>,<rpm>,<count>,<s0>,<s1>,...,<sN-1>\n`

### 3.2. `src/features.py`
- **Strict Input Validation**: Added `validate_window()` ensuring inputs are non-empty, 1-dimensional, finite, and have at least 4 samples.
- **Hann Windowing**: Applied `np.hanning()` after mean subtraction to minimize spectral sidelobe leakage.
- **Zero & Constant Signal Protection**: Detected zero-variance and flatline conditions (`arr_std < 1e-12`). Returns `kurtosis = 0.0`, `dominant_freq = 0.0`, and `spectral_energy = 0.0` instead of `NaN`.
- **DC Exclusion**: Excluded the 0 Hz DC bin from dominant frequency search to prevent sensor bias/gravity offset from masking rotor fundamental frequency.
- **Reusable Spectrum**: Added `return_spectrum=True` option so single-pass FFT can be passed directly to the dashboard without recomputation.

### 3.3. `src/calibration.py`
- **Feature-Specific Scale Floors**: Added `DEFAULT_SCALE_FLOORS = [0.01, 2.0, 100.0, 0.1]` and relative scale floors (`0.05 * abs(mean)`). Effective scale is computed as:
  $$\text{effective\_scale} = \max(\text{std}, \text{scale\_floor}, 0.05 \cdot |\text{mean}|)$$
  This guarantees that steady-state features (e.g. constant dominant frequency across calibration windows where $\text{std} = 0$) do not blow up into multi-million z-scores on fractional frequency shifts.
- **Window Validation**: Enforces minimum number of calibration windows ($\ge 3$), checks for non-zero RMS to reject disconnected sensors, and validates numerical finiteness.
- **Atomic Persistence**: Implemented atomic JSON writes (`.tmp` + `os.replace`) to prevent corrupted files on unexpected shutdown.

### 3.4. `src/dataset.py`, `src/train.py`, and `src/ablation.py`
- **Sampling Rate Consistency**: Aligned synthetic generation to the hardware-feasible sample rate of 800 Hz (Nyquist: 400 Hz).
- **Physical Signal Synthesis**: Modeled 1X rotational unbalance centrifugal force, 2X harmonic distortion, and Gaussian background noise.
- **Evaluation Metrics**: Upgraded evaluation in `train.py` to report Confusion Matrix, False Positive Rate (FPR), False Negative Rate (FNR), Normal-class Precision/Recall/F1, and Fault-class Precision/Recall/F1.
- **Ablation Verification**: Verified in `ablation.py` that adaptive baseline calibration achieves 100% accuracy on unseen deployments versus 83.3% for fixed global scaling.

### 3.5. `serving/app.py`
- **Validation & Error Handling**: Enforces JSON validation, finite checks, and sample count limits on `/ingest`. Catches all processing exceptions and returns HTTP 400 with descriptive error messages instead of HTTP 500 crashes.
- **Eliminated Redundant FFT**: Reuses single-pass FFT returned by `extract_features(..., return_spectrum=True)`.
- **Connection Diagnostics**: Tracks packet timestamps; flags `is_stale: true` when data has not arrived for $> 3.0$ seconds.
- **Motor Command Queue**: Added `/motor/control` and `/motor/pending_command` endpoints to enable UI-driven motor management.
- **Local Security**: Disabled external interface binding and set `debug=False` for local deployment.

### 3.6. `serving/serial_reader.py`
- **Packet Parser**: Implemented robust parsing for `$VIB` packets, rejecting length mismatches, incomplete headers, or malformed numeric fields.
- **Resilience**: Added auto-reconnect loop on `serial.SerialException` (cable unplugged/replugged). Added `--list-ports` utility to enumerate available COM ports on Windows.
- **Bidirectional Control**: Polls Flask `/motor/pending_command` every 200 ms and writes commands to the Arduino serial port.

### 3.7. `serving/build_demo_cache.py` and `serving/simulate_stream.py`
- **Autonomous Fallback**: Added automatic synthetic vibration synthesis at 800 Hz when raw MaFaulDa CSV files are absent, allowing the local simulation to run immediately out-of-the-box.
- **Simulated Tachometer**: Included RPM telemetry in simulated streams to verify dashboard tachometer behavior.

### 3.8. `dashboard/templates/index.html` and `dashboard/static/dashboard.js`
- **Aesthetic**: Refactored interface into a modern dark-mode industrial diagnostic view using Inter and JetBrains Mono typography.
- **Stale Data Warning**: Displays an amber warning banner if packets stop arriving for $>3$ seconds.
- **Status & Confidence**: Explicitly labels classification as `Model Confidence (Fault Probability)` to prevent misrepresenting statistical confidence as physical machine health.
- **Motor Control Widgets**: Added interactive Start/Stop buttons and a PWM speed slider (0-255).

---

## 4. Test Execution and Verification Results

### 4.1. Arduino Uno Firmware Compilation Test
- **Tool**: `arduino-cli` v1.8.6
- **Target**: `arduino:avr:uno` (ATmega328P)
- **Result**: **PASSED** (Exit code: 0)
- **Program Memory (Flash)**: 8,420 bytes (26% of 32,256 bytes)
- **Dynamic Memory (SRAM)**: 1,029 bytes (50% of 2,048 bytes)
- **Free Headroom**: 1,019 bytes (50% dynamic headroom)

### 4.2. Automated Python Test Suite (`tests/test_pipeline.py`)
- **Execution Command**: `python tests/test_pipeline.py -v`
- **Total Tests Run**: 17
- **Passed**: 17
- **Failures / Errors**: 0
- **Summary**:
  - `test_zero_signal`: Passed (all features 0, no NaN).
  - `test_constant_signal`: Passed (RMS correct, kurtosis 0, no NaN).
  - `test_known_sine_frequency`: Passed (detected 35 Hz sine at 800 Hz sample rate within $\pm 1.5$ Hz).
  - `test_invalid_and_nan_inputs`: Passed (caught None, empty, NaN, Inf, short windows).
  - `test_different_window_sizes`: Passed (verified 64, 128, 256, 512, 1024 sample arrays).
  - `test_zero_variance_dominant_frequency`: Passed (effective scale floor $\ge 2.0$, z-score bounded).
  - `test_minimum_windows_validation`: Passed (rejected $<3$ windows).
  - `test_flatlined_sensor_rejection`: Passed (rejected zero RMS calibration windows).
  - `test_save_load_round_trip`: Passed (perfect parameter recovery).
  - `test_health_endpoint`: Passed (HTTP 200).
  - `test_ingest_validation`: Passed (rejected empty/malformed/NaN payloads with HTTP 400).
  - `test_calibration_lifecycle`: Passed (validated start, accumulate, stop, fit).
  - `test_motor_control_endpoints`: Passed (validated START, STOP, and command queueing).
  - `test_valid_packet`: Passed (parsed `$VIB,42,800,1800.0,4,...`).
  - `test_count_mismatch`: Passed (rejected mismatched packet counts).
  - `test_malformed_numeric_data`: Passed (rejected non-numeric values).
  - `test_status_messages`: Passed (handled `$STATUS` and `$ERR` cleanly).

### 4.3. End-to-End Simulation Test (`serving/simulate_stream.py`)
- **Backend**: Flask API listening on `127.0.0.1:5000`
- **Stream**: 10 normal calibration windows, 15 unseen normal evaluation windows, 25 imbalance fault windows.
- **Results**:
  - Calibration: Successfully fitted on 10 windows (`status: calibrated`).
  - Normal Windows 1–15: **15/15 classified as `normal`** (Model fault confidence: 2.1% to 13.9%).
  - Imbalance Windows 1–25: **25/25 classified as `imbalance`** (Model fault confidence: 100.0%).

---

## 5. Remaining Limitations and Hardware Verification Status

The software, signal processing algorithms, serial communications, and embedded firmware have been verified through static analysis, build verification, and automated testing. However, the following physical properties require human confirmation on the physical test bench:

1. **Physical DC Motor Supply Voltage & Current Rating**:
   The hardware BOM specifies a 12 V DC, 1 A adapter. A 555 brushed DC motor under load or stall can draw significantly more than 1.0 A (often 2 A to 5 A peak). The power supply must be verified against the motor's actual stall current before running at high PWM duty cycles to avoid tripping adapter overcurrent protection.
2. **Optical Speed Sensor Slot Count**:
   The firmware assumes a standard 20-slot optical encoder disc (`PULSES_PER_REV = 20`). If the physical rig uses a different slot count or a single reflective tab, update `PULSES_PER_REV` in `firmware/vibration_monitor.ino`.
3. **Mechanical Safety Guarding**:
   Rotor unbalance testing involves attaching eccentric masses to a rotating motor disc. Never spin the disc with test weights without a secure mechanical enclosure or safety shield.
4. **Physical COM Port Assignment**:
   When connecting the Arduino Uno to a Windows laptop, determine the assigned port in Windows Device Manager (e.g., `COM3`, `COM4`) before launching `serial_reader.py`.
