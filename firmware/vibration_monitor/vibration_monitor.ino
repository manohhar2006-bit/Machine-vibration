/*
 * Predictive Machine Vibration Monitoring Firmware
 * Target Microcontroller: Arduino Uno R3 (ATmega328P, 16 MHz, 2 KB SRAM)
 * 
 * Hardware Subsystems:
 *  1. Vibration Sensor: ADXL345 digital 3-axis accelerometer on I2C bus (A4=SDA, A5=SCL).
 *     - Output Data Rate (ODR): 800 Hz (BW_RATE = 0x0D, 400 Hz analog bandwidth).
 *     - Full resolution mode (+/-2g range, 3.9 mg/LSB scale factor).
 *     - I2C Fast Mode (400 kHz clock).
 * 
 *  2. Rotational Speed Sensor: IR slotted optical sensor on Pin 2 (INT0).
 *     - External interrupt with atomic pulse read.
 *     - Configurable pulses-per-revolution (PPR).
 * 
 *  3. Motor Driver: L298N H-Bridge Driver.
 *     - ENA: Pin 9 (Timer 1 PWM speed control).
 *     - IN1: Pin 8 (Direction control).
 *     - IN2: Pin 7 (Direction control).
 *     - Safety: Defaults strictly to STOPPED / DISABLED on boot or reset.
 * 
 * Memory Budget (2048 bytes SRAM total on ATmega328P):
 *  - accelBuffer: 256 samples * 2 bytes (int16_t) = 512 bytes (25.0% SRAM)
 *  - Wire I2C buffers: 5 * 32 bytes = 160 bytes
 *  - Serial TX/RX ring buffers: 64 + 64 = 128 bytes
 *  - Command line buffer & runtime variables: ~80 bytes
 *  - Total static allocation: ~880 bytes (~43% of SRAM)
 *  - Dynamic headroom for stack: > 1,150 bytes (> 56% free)
 * 
 * Communication Protocol:
 *  - Serial UART at 115200 baud.
 *  - Vibration Window Packet:
 *      $VIB,<seq>,<sample_rate>,<rpm>,<count>,<s0>,<s1>,...,<sN-1>\n
 *  - Diagnostic / Status Packets:
 *      $STATUS,SENSOR=<OK|ERR>,MOTOR=<STOPPED|RUNNING>,SPEED=<0-255>,RPM=<rpm>\n
 *      $OK,<message>\n
 *      $ERR,<message>\n
 *  - Supported Serial Commands (case-insensitive):
 *      MOTOR START [speed]   -- Start motor (optional speed 0-255, default 180)
 *      MOTOR STOP            -- Stop and disable motor driver
 *      MOTOR SPEED <0-255>   -- Set PWM duty cycle
 *      STATUS                -- Report subsystem status
 */

#include <Wire.h>

// -----------------------------------------------------------------------------
// Pin Configuration
// -----------------------------------------------------------------------------
const uint8_t PIN_IR_SENSOR  = 2;  // External interrupt 0 (INT0)
const uint8_t PIN_MOTOR_IN2  = 7;  // L298N direction pin 2
const uint8_t PIN_MOTOR_IN1  = 8;  // L298N direction pin 1
const uint8_t PIN_MOTOR_ENA  = 9;  // L298N PWM speed enable (Timer 1)

// -----------------------------------------------------------------------------
// ADXL345 Registers & Settings
// -----------------------------------------------------------------------------
#define ADXL345_ADDR        0x53  // I2C address when ALT ADDRESS pin is GND
#define ADXL345_DEVID_REG   0x00  // Device ID register (expects 0xE5)
#define ADXL345_BW_RATE     0x2C  // Data rate and power mode control
#define ADXL345_POWER_CTL   0x2D  // Power-saving features control
#define ADXL345_DATA_FORMAT 0x31  // Data format control
#define ADXL345_DATAX0      0x32  // X-axis data 0 (LSB)

// Sampling parameters
const unsigned long SAMPLE_RATE_HZ     = 800;   // 800 Hz ODR matches ADXL345 BW_RATE 0x0D
const unsigned long SAMPLE_INTERVAL_US = 1250;  // 1 / 800 Hz = 1250 microseconds
const uint16_t      WINDOW_SIZE        = 256;   // 256 samples = 0.320 s window

// Buffer allocation: stored as raw int16_t to strictly fit within ATmega328P SRAM
int16_t  accelBuffer[WINDOW_SIZE];
uint16_t bufferIndex = 0;
uint16_t packetSequence = 0;

// Hardware state
bool sensorOnline = false;
bool motorRunning = false;
uint8_t motorPwmSpeed = 180;  // Default safe initial operating speed (~70% duty cycle)

// -----------------------------------------------------------------------------
// IR Optical Speed Sensor Configuration
// -----------------------------------------------------------------------------
// Standard optical encoder disc with 20 slots. Change to 1 if using single reflective tab.
const unsigned int PULSES_PER_REV = 20;

volatile unsigned long pulseCount = 0;
unsigned long lastRpmTimeMs = 0;
float currentRpm = 0.0f;

// Interrupt Service Routine for IR Sensor
void irSensorIsr() {
  pulseCount++;
}

// -----------------------------------------------------------------------------
// Serial Command Line Buffer
// -----------------------------------------------------------------------------
char cmdBuffer[32];
uint8_t cmdIndex = 0;

// -----------------------------------------------------------------------------
// I2C Helper Functions
// -----------------------------------------------------------------------------
bool adxlWriteRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

uint8_t adxlReadRegister(uint8_t reg) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom((uint8_t)ADXL345_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

int16_t adxlReadRawX() {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(ADXL345_DATAX0);
  if (Wire.endTransmission(false) != 0) return 0;

  Wire.requestFrom((uint8_t)ADXL345_ADDR, (uint8_t)2);
  if (Wire.available() >= 2) {
    uint8_t lsb = Wire.read();
    uint8_t msb = Wire.read();
    return (int16_t)((msb << 8) | lsb);
  }
  return 0;
}

// -----------------------------------------------------------------------------
// Motor Driver Control Functions
// -----------------------------------------------------------------------------
void motorInit() {
  pinMode(PIN_MOTOR_ENA, OUTPUT);
  pinMode(PIN_MOTOR_IN1, OUTPUT);
  pinMode(PIN_MOTOR_IN2, OUTPUT);

  // Safety Requirement: Default strictly to STOPPED / DISABLED
  analogWrite(PIN_MOTOR_ENA, 0);
  digitalWrite(PIN_MOTOR_IN1, LOW);
  digitalWrite(PIN_MOTOR_IN2, LOW);
  motorRunning = false;
}

void motorStart(uint8_t speed) {
  motorPwmSpeed = speed;
  digitalWrite(PIN_MOTOR_IN1, HIGH);
  digitalWrite(PIN_MOTOR_IN2, LOW);
  analogWrite(PIN_MOTOR_ENA, motorPwmSpeed);
  motorRunning = true;
  Serial.print(F("$OK,MOTOR_STARTED,SPEED="));
  Serial.println(motorPwmSpeed);
}

void motorStop() {
  analogWrite(PIN_MOTOR_ENA, 0);
  digitalWrite(PIN_MOTOR_IN1, LOW);
  digitalWrite(PIN_MOTOR_IN2, LOW);
  motorRunning = false;
  Serial.println(F("$OK,MOTOR_STOPPED"));
}

void motorSetSpeed(uint8_t speed) {
  motorPwmSpeed = speed;
  if (motorRunning) {
    analogWrite(PIN_MOTOR_ENA, motorPwmSpeed);
  }
  Serial.print(F("$OK,MOTOR_SPEED="));
  Serial.println(motorPwmSpeed);
}

// -----------------------------------------------------------------------------
// RPM Calculation (Atomic Pulse Read)
// -----------------------------------------------------------------------------
void updateRpm() {
  unsigned long nowMs = millis();
  unsigned long elapsedMs = nowMs - lastRpmTimeMs;

  // Update RPM every 250 ms for smooth and responsive tracking
  if (elapsedMs >= 250) {
    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    float minutes = (float)elapsedMs / 60000.0f;
    float revolutions = (float)pulses / (float)PULSES_PER_REV;
    currentRpm = (minutes > 0.0f) ? (revolutions / minutes) : 0.0f;
    lastRpmTimeMs = nowMs;
  }
}

// -----------------------------------------------------------------------------
// Serial Command Processor
// -----------------------------------------------------------------------------
void processCommand(const char* cmd) {
  // Trim leading whitespace
  while (*cmd == ' ' || *cmd == '\t') cmd++;

  if (strncmp(cmd, "MOTOR START", 11) == 0 || strncmp(cmd, "START", 5) == 0) {
    const char* p = cmd + (strncmp(cmd, "MOTOR START", 11) == 0 ? 11 : 5);
    while (*p == ' ') p++;
    int spd = (*p != '\0') ? atoi(p) : motorPwmSpeed;
    if (spd < 0) spd = 0;
    if (spd > 255) spd = 255;
    motorStart((uint8_t)spd);

  } else if (strcmp(cmd, "MOTOR STOP") == 0 || strcmp(cmd, "STOP") == 0) {
    motorStop();

  } else if (strncmp(cmd, "MOTOR SPEED", 11) == 0 || strncmp(cmd, "SPEED", 5) == 0) {
    const char* p = cmd + (strncmp(cmd, "MOTOR SPEED", 11) == 0 ? 11 : 5);
    while (*p == ' ') p++;
    int spd = atoi(p);
    if (spd < 0) spd = 0;
    if (spd > 255) spd = 255;
    motorSetSpeed((uint8_t)spd);

  } else if (strcmp(cmd, "STATUS") == 0) {
    Serial.print(F("$STATUS,SENSOR="));
    Serial.print(sensorOnline ? F("OK") : F("DISCONNECTED"));
    Serial.print(F(",MOTOR="));
    Serial.print(motorRunning ? F("RUNNING") : F("STOPPED"));
    Serial.print(F(",SPEED="));
    Serial.print(motorPwmSpeed);
    Serial.print(F(",RPM="));
    Serial.println(currentRpm, 1);

  } else if (strlen(cmd) > 0) {
    Serial.print(F("$ERR,UNKNOWN_COMMAND:"));
    Serial.println(cmd);
  }
}

void checkSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = '\0';
        // Convert to uppercase for case-insensitive matching
        for (uint8_t i = 0; i < cmdIndex; i++) {
          if (cmdBuffer[i] >= 'a' && cmdBuffer[i] <= 'z') {
            cmdBuffer[i] = cmdBuffer[i] - 'a' + 'A';
          }
        }
        processCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else if (cmdIndex < (sizeof(cmdBuffer) - 1)) {
      cmdBuffer[cmdIndex++] = c;
    }
  }
}

// -----------------------------------------------------------------------------
// Transmit Vibration Window Packet
// -----------------------------------------------------------------------------
void transmitWindow() {
  // Format: $VIB,<seq>,<sample_rate>,<rpm>,<count>,<val0>,<val1>,...
  Serial.print(F("$VIB,"));
  Serial.print(packetSequence++);
  Serial.print(F(","));
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(F(","));
  Serial.print(currentRpm, 1);
  Serial.print(F(","));
  Serial.print(WINDOW_SIZE);

  // Transmit values scaled to acceleration in g (3.9 mg/LSB = 0.0039 g/LSB)
  for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
    Serial.print(F(","));
    float g_val = (float)accelBuffer[i] * 0.0039f;
    Serial.print(g_val, 3);
  }
  Serial.println();
}

// -----------------------------------------------------------------------------
// Arduino Setup
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 1000);  // Brief wait for native USB, if applicable

  Serial.println(F("$BOOT,Predictive Machine Vibration Monitoring Uno R3 Initializing..."));

  // 1. Initialize motor driver in safe disabled state
  motorInit();

  // 2. Initialize IR speed sensor
  pinMode(PIN_IR_SENSOR, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_IR_SENSOR), irSensorIsr, FALLING);
  lastRpmTimeMs = millis();

  // 3. Initialize I2C bus at 400 kHz Fast Mode
  Wire.begin();
  Wire.setClock(400000);

  // 4. Check ADXL345 connection and device ID
  uint8_t devId = adxlReadRegister(ADXL345_DEVID_REG);
  if (devId == 0xE5) {
    sensorOnline = true;

    // Configure Data Rate: 800 Hz ODR (BW_RATE = 0x0D)
    adxlWriteRegister(ADXL345_BW_RATE, 0x0D);

    // Configure Data Format: Full resolution mode, +/-2g range (DATA_FORMAT = 0x08)
    adxlWriteRegister(ADXL345_DATA_FORMAT, 0x08);

    // Set measurement mode (POWER_CTL = 0x08)
    adxlWriteRegister(ADXL345_POWER_CTL, 0x08);

    Serial.println(F("$OK,ADXL345_INITIALIZED_800HZ"));
  } else {
    sensorOnline = false;
    Serial.print(F("$ERR,ADXL345_NOT_FOUND,DEVID="));
    Serial.println(devId, HEX);
  }

  Serial.println(F("$READY,Waiting for commands or sampling..."));
}

// -----------------------------------------------------------------------------
// Arduino Main Loop
// -----------------------------------------------------------------------------
void loop() {
  static unsigned long lastSampleUs = 0;
  unsigned long nowUs = micros();

  // 1. Check for incoming motor control and status commands
  checkSerialInput();

  // 2. Update RPM estimate from optical sensor interrupts
  updateRpm();

  // 3. Fixed sample interval scheduling (rollover-safe unsigned subtraction)
  if (nowUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
    lastSampleUs = nowUs;

    if (sensorOnline) {
      accelBuffer[bufferIndex++] = adxlReadRawX();
    } else {
      // Missing sensor fallback: do not fabricate vibrations
      accelBuffer[bufferIndex++] = 0;
    }

    if (bufferIndex >= WINDOW_SIZE) {
      transmitWindow();
      bufferIndex = 0;
    }
  }
}
