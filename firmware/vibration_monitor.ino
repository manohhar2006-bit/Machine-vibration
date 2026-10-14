/*
 * Vibration monitor firmware -- Arduino Uno.
 *
 * Reads the ADXL345 accelerometer over I2C and an IR slotted speed
 * sensor via a hardware interrupt, samples on a fixed micros()-timed
 * interval, buffers WINDOW_SIZE samples, then transmits the buffered
 * accelerometer values plus the measured RPM as one CSV line over
 * Serial:
 *     <rpm>,<sample_1>,<sample_2>,...,<sample_N>
 *
 * Matches the packet format expected by serving/serial_reader.py.
 *
 * Untested against physical hardware; written to the ADXL345 / IR
 * speed sensor specification, not yet verified on the assembled rig.
 */
#include <Wire.h>

#define ADXL345_ADDR 0x53
#define ADXL345_POWER_CTL 0x2D
#define ADXL345_DATAX0 0x32

const int IR_SENSOR_PIN = 2;                   // must support external interrupts
const unsigned long SAMPLE_INTERVAL_US = 250;  // 4000 Hz
const int WINDOW_SIZE = 4000;                  // 1 second at 4000 Hz

float accelBuffer[WINDOW_SIZE];
int bufferIndex = 0;

volatile unsigned long pulseCount = 0;
unsigned long lastRpmCalcMs = 0;

void countPulse() {
  pulseCount++;
}

void adxlWriteRegister(byte reg, byte value) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  adxlWriteRegister(ADXL345_POWER_CTL, 0x08);  // measurement mode

  pinMode(IR_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(IR_SENSOR_PIN), countPulse, FALLING);

  lastRpmCalcMs = millis();
}

float readAccelX() {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(ADXL345_DATAX0);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 2);

  int16_t raw = Wire.read() | (Wire.read() << 8);
  return raw * 0.0039;  // 3.9 mg/LSB at the default +-2g range
}

void loop() {
  static unsigned long lastSampleUs = 0;
  unsigned long nowUs = micros();

  if (nowUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
    lastSampleUs = nowUs;

    accelBuffer[bufferIndex] = readAccelX();
    bufferIndex++;

    if (bufferIndex >= WINDOW_SIZE) {
      unsigned long nowMs = millis();
      float elapsedS = (nowMs - lastRpmCalcMs) / 1000.0;
      float rpm = (pulseCount / elapsedS) * 60.0;  // one pulse per revolution
      pulseCount = 0;
      lastRpmCalcMs = nowMs;

      Serial.print(rpm, 2);
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Serial.print(',');
        Serial.print(accelBuffer[i], 4);
      }
      Serial.println();

      bufferIndex = 0;
    }
  }
}
