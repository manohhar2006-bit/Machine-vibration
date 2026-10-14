# Hardware Wiring & Electrical Guide
## Predictive Machine Vibration Monitoring Rig

**Target Microcontroller:** Arduino Uno R3 (ATmega328P)  
**Actuators & Sensors:** 555 Brushed DC Motor, L298N Dual H-Bridge Driver, ADXL345 3-Axis Digital Accelerometer, Slotted Optical Speed Sensor (IR Tachometer).

---

## 1. Master Pin Assignment Summary

| Subsystem | Component Pin | Arduino Uno Pin | Pin Type / Function | Notes |
|---|---|---|---|---|
| **I2C Bus** | ADXL345 `SDA` | **A4** | Hardware I2C Data | Connect 4.7 k&Omega; pullup to 3.3V if breakout module does not include one |
| **I2C Bus** | ADXL345 `SCL` | **A5** | Hardware I2C Clock | Hardware Fast Mode (400 kHz) |
| **I2C Power** | ADXL345 `VCC` | **3.3V** or **5V** | Power (Check breakout) | Most breakouts include 3.3V onboard LDO regulator; check board marking |
| **I2C Ground**| ADXL345 `GND` | **GND** | System Ground | Common ground |
| **I2C Config**| ADXL345 `CS` | **3.3V** (or VCC) | Bus Select | High selects I2C mode (not SPI) |
| **I2C Config**| ADXL345 `SDO / ALT`| **GND** | Address Select | Low selects I2C address `0x53` (High selects `0x54`) |
| **Tachometer**| IR Sensor `OUT` | **Pin 2** | Digital Input (INT0) | External interrupt 0; uses `INPUT_PULLUP` |
| **Tachometer**| IR Sensor `VCC` | **5V** | Sensor Power | From Arduino 5V pin |
| **Tachometer**| IR Sensor `GND` | **GND** | Sensor Ground | Common ground |
| **Motor Driver**| L298N `ENA` | **Pin 9** | PWM Output (Timer 1) | Speed control (remove jumper cap on ENA pin first!) |
| **Motor Driver**| L298N `IN1` | **Pin 8** | Digital Output | Direction control: HIGH for Forward |
| **Motor Driver**| L298N `IN2` | **Pin 7** | Digital Output | Direction control: LOW for Forward |
| **Motor Driver**| L298N `GND` | **GND** | Logic / Power Ground | **MANDATORY**: Must tie to Arduino GND and Power Adapter (-) |
| **Motor Power** | L298N `12V / VMS`| **Adapter (+)** | Motor Power Input | 12 V DC Power Supply (+) terminal |
| **Motor Power** | L298N `GND` | **Adapter (-)** | Motor Power Return | 12 V DC Power Supply (-) terminal |
| **Motor Leads** | Motor Terminal A | L298N `OUT1` | Motor Output A | Screw terminal on L298N |
| **Motor Leads** | Motor Terminal B | L298N `OUT2` | Motor Output B | Screw terminal on L298N |

---

## 2. Electrical Wiring Schematics & Common Ground

```
  +-------------------------------------------------------------------------+
  |                          COMMON GROUND TOPOLOGY                         |
  |                                                                         |
  |   +-----------------------+              +--------------------------+   |
  |   |   12V DC Adapter      |              |      Arduino Uno R3      |   |
  |   |  (+)             (-)  |              |                          |   |
  |   +---|---------------|---+              |       GND        5V  3.3V|   |
  |       |               |                  +--------|----------|----|-+   |
  |       | +12V          | GND                       |          |    |     |
  |       |               +---------------------------+          |    |     |
  |       |               |                                      |    |     |
  |       |               | (COMMON GND)                         |    |     |
  |       |               |                                      |    |     |
  |   +---|---------------|-------------------+                  |    |     |
  |   |  12V             GND             5V   |                  |    |     |
  |   |                                       |                  |    |     |
  |   |       L298N Dual H-Bridge             |                  |    |     |
  |   |  ENA     IN1     IN2     OUT1   OUT2  |                  |    |     |
  |   +---|-------|-------|--------|------|---+                  |    |     |
  |       |       |       |        |      |                      |    |     |
  |       | D9    | D8    | D7     +------+---> [555 DC Motor]   |    |     |
  |       |       |       |                                      |    |     |
  |       +-------+-------+--------------------------------------+    |     |
  |                                                                   |     |
  |   +-------------------------+                                     |     |
  |   | IR Speed Sensor Module  |                                     |     |
  |   |   VCC     GND     OUT   |                                     |     |
  |   +----|-------|-------|----+                                     |     |
  |        |       |       +------------------> Pin 2 (INT0)          |     |
  |        |       +--------------------------> Arduino GND           |     |
  |        +----------------------------------> Arduino 5V            |     |
  |                                                                   |     |
  |   +------------------------------------+                          |     |
  |   | ADXL345 Accelerometer Module       |                          |     |
  |   |   VCC   GND   SDA   SCL   CS   SDO |                          |     |
  |   +----|-----|-----|-----|-----|----|--+                          |     |
  |        |     |     |     |     |    +-----> Arduino GND (0x53)    |     |
  |        |     |     |     |     +----------> 3.3V (Select I2C)     |     |
  |        |     |     |     +----------------> Pin A5 (SCL)          |     |
  |        |     |     +----------------------> Pin A4 (SDA)          |     |
  |        |     +----------------------------> Arduino GND           |     |
  |        +----------------------------------> Arduino 3.3V / 5V     |     |
  +-------------------------------------------------------------------------+
```

---

## 3. Detailed Subsystem Wiring Instructions

### 3.1. ADXL345 Accelerometer Wiring (I2C)
1. **Mechanical Mounting**: Mount the ADXL345 firmly to the motor housing or bearing bracket using a rigid screw or industrial double-sided acrylic tape. The measurement axis in firmware is **X-axis (`DATAX0/1`)**. Orient the sensor so the X-axis points in the **radial direction** of motor vibration (perpendicular to the motor shaft).
2. **I2C Communication Pins**:
   - Connect **SDA** to Arduino **A4**.
   - Connect **SCL** to Arduino **A5**.
3. **Module Address Configuration**:
   - Connect **SDO / ALT ADDRESS** to **GND**. This sets the 7-bit I2C device address to `0x53` (matching the `#define ADXL345_ADDR 0x53` in firmware).
   - Connect **CS** to **VCC** (or 3.3V). This puts the ADXL345 in I2C communication mode.
4. **Power**:
   - If using the standard blue GY-291 breakout board, it features an onboard RT9183 3.3V regulator and can accept 3.3V or 5V on `VCC`.

### 3.2. IR Optical Speed Sensor Wiring (Tachometer)
1. **Sensor Placement**: Align the slotted optical sensor so that the slotted disc attached to the motor shaft passes cleanly through the optical gap without mechanical rubbing.
2. **Connections**:
   - `VCC` &rarr; Arduino **5V**
   - `GND` &rarr; Arduino **GND**
   - `OUT / D0` &rarr; Arduino **Pin 2** (ATmega328P external interrupt `INT0`).
3. **PPR Verification**: Count the number of slots or holes in the optical disc. The default firmware setting is:
   ```cpp
   const unsigned int PULSES_PER_REV = 20;
   ```
   If using a disc with 1 slot or a single reflective flag, change this value to `1`.

### 3.3. L298N Motor Driver Wiring
1. **Remove ENA Jumper**: The L298N module usually ships with a black jumper cap connecting the `ENA` pin to 5V (forcing 100% duty cycle). **Remove this jumper cap** so the Arduino can control speed via PWM on Pin 9.
2. **Logic Control Pins**:
   - Connect **ENA** to Arduino **Pin 9** (Timer 1 PWM).
   - Connect **IN1** to Arduino **Pin 8**.
   - Connect **IN2** to Arduino **Pin 7**.
3. **Motor Leads**:
   - Connect the two terminals of the 555 brushed DC motor to **OUT1** and **OUT2** screw terminals on the L298N.
4. **Logic Power Jumper (5V Enable)**:
   - If powering the L298N with 12 V on the `VMS` terminal, the onboard 78M05 5V regulator jumper may remain installed. This supplies 5V to the internal logic of the L298N.

---

## 4. Critical Electrical Safety & Verification Rules

> [!CAUTION]
> **COMMON GROUND REQUIREMENT:** The Arduino Uno GND terminal, the L298N GND terminal, and the negative (-) lead of the external 12 V power adapter **MUST be tied together**. If the common ground connection is missing, logic signals will have no return path, causing motor runaway, erratic PWM switching, or false sensor readings.

> [!WARNING]
> **NEVER POWER MOTOR FROM ARDUINO 5V PIN:**
> The Arduino Uno onboard 5V regulator can supply a maximum of 400 mA to 500 mA. A 555 DC motor draws 300 mA to 800 mA running unloaded and 2.0 A to 5.0 A when stalled or starting up. Attempting to power the motor from the Arduino 5V or 3.3V header will destroy the microcontroller's voltage regulator or cause immediate continuous USB brownout resets.

> [!IMPORTANT]
> **POWER ADAPTER CURRENT RATING VERIFICATION:**
> The project specifications mention a 12 V, 1.0 A DC power adapter.
> - **Verify your 555 motor nameplate**: Typical 555 motors have stall currents ranging from 1.5 A to 4.5 A.
> - If your motor stall current exceeds 1.0 A, starting the motor at 100% duty cycle (`speed = 255`) may trigger the power adapter's internal overcurrent trip.
> - **Safe Start Procedure**: Always start the motor at a moderate PWM duty cycle (`speed = 150` to `180`) rather than full 255 to minimize startup inrush current.
