/*
 * Vibration monitor firmware -- Arduino Uno.
 *
 * Drives the DC motor through the L298N, reads the ADXL345
 * accelerometer over I2C and an IR slotted speed sensor via a hardware
 * interrupt. Samples on a fixed micros()-timed interval, buffers
 * WINDOW_SIZE samples, then transmits the mean-removed acceleration (g)
 * plus the measured RPM as one CSV line over Serial:
 *     <rpm>,<sample_1>,<sample_2>,...,<sample_N>
 *
 * Matches the packet format expected by serving/serial_reader.py
 * (SAMPLE_RATE and WINDOW_SIZE there must match the values below).
 *
 * ADXL345 wiring (breadboard, bottom rails):
 *   GND  -> bottom blue (GND) rail
 *   VCC  -> bottom red rail, 3.3V
 *   CS   -> bottom red rail, 3.3V (tied high selects I2C mode)
 *   SDO  -> bottom blue (GND) rail (selects I2C address 0x53)
 *   SDA  -> Arduino A4
 *   SCL  -> Arduino A5
 *   INT1, INT2 -> not connected
 *
 * IR speed sensor: signal -> D2 (external interrupt), power from the
 * top 5V rail.
 *
 * L298N: ENA -> D10 (PWM speed), IN1 -> D8, IN2 -> D9. Motor supply on
 * the L298N is a separate supply sharing a common ground with the Uno.
 *
 * Memory: the Uno has 2 KB of SRAM, so samples are stored as raw int16
 * values (2 bytes each) and converted to g only when transmitted.
 */
#include <Wire.h>

#define ADXL345_ADDR 0x53
#define ADXL345_DEVID 0x00
#define ADXL345_BW_RATE 0x2C
#define ADXL345_POWER_CTL 0x2D
#define ADXL345_DATA_FORMAT 0x31
#define ADXL345_DATAX0 0x32
#define ADXL345_DEVID_EXPECTED 0xE5
#define ADXL345_ODR_1600HZ 0x0E  // output data rate, above the sampling rate
#define ADXL345_MEASURE 0x08
#define ADXL345_FULL_RES_2G 0x08  // full resolution, +-2g, 3.9 mg/LSB

const byte IR_SENSOR_PIN = 2;  // must support external interrupts
const byte MOTOR_ENA_PIN = 10;  // PWM capable
const byte MOTOR_IN1_PIN = 8;
const byte MOTOR_IN2_PIN = 9;
const byte MOTOR_PWM_DUTY = 180;  // 0-255, motor speed

const unsigned long SAMPLE_INTERVAL_US = 1000;  // 1000 Hz
const int WINDOW_SIZE = 512;
const float PULSES_PER_REV = 1.0;  // slots on the speed disc
const float G_PER_LSB = 0.0039;

int16_t accelBuffer[WINDOW_SIZE];
int bufferIndex = 0;
long bufferSum = 0;

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

byte adxlReadRegister(byte reg) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 1);
  return Wire.read();
}

// Confirms the ADXL345 responds with its fixed device ID (0xE5) at 0x53.
// On failure, halts and reports over Serial instead of streaming a
// meaningless signal. The cause is almost always wiring.
void checkAdxlPresent() {
  byte devid = adxlReadRegister(ADXL345_DEVID);

  if (devid != ADXL345_DEVID_EXPECTED) {
    Serial.print(F("ADXL345 not found at 0x53 (DEVID read: 0x"));
    Serial.print(devid, HEX);
    Serial.println(F("). Check VCC/GND/CS/SDO wiring and SDA/SCL connections."));
    while (true) {
    }
  }

  Serial.println(F("ADXL345 found at 0x53."));
}

void startMotor() {
  pinMode(MOTOR_ENA_PIN, OUTPUT);
  pinMode(MOTOR_IN1_PIN, OUTPUT);
  pinMode(MOTOR_IN2_PIN, OUTPUT);
  digitalWrite(MOTOR_IN1_PIN, HIGH);
  digitalWrite(MOTOR_IN2_PIN, LOW);
  analogWrite(MOTOR_ENA_PIN, MOTOR_PWM_DUTY);
}

int16_t readRawX() {
  static int16_t lastRaw = 0;

  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(ADXL345_DATAX0);
  Wire.endTransmission(false);

  if (Wire.requestFrom(ADXL345_ADDR, 2) < 2) {
    return lastRaw;  // failed read, repeat the previous sample
  }

  lastRaw = Wire.read() | (Wire.read() << 8);
  return lastRaw;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);  // fast mode, needed to read every 1 ms reliably

  checkAdxlPresent();

  adxlWriteRegister(ADXL345_DATA_FORMAT, ADXL345_FULL_RES_2G);
  adxlWriteRegister(ADXL345_BW_RATE, ADXL345_ODR_1600HZ);
  adxlWriteRegister(ADXL345_POWER_CTL, ADXL345_MEASURE);

  pinMode(IR_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(IR_SENSOR_PIN), countPulse, FALLING);

  startMotor();

  lastRpmCalcMs = millis();
}

void sendWindow() {
  noInterrupts();
  unsigned long pulses = pulseCount;
  pulseCount = 0;
  interrupts();

  unsigned long nowMs = millis();
  float elapsedS = (nowMs - lastRpmCalcMs) / 1000.0;
  lastRpmCalcMs = nowMs;
  float rpm = (pulses / PULSES_PER_REV / elapsedS) * 60.0;

  // Removing the window mean strips the static gravity component so the
  // RMS feature reflects vibration only.
  float mean = (float)bufferSum / WINDOW_SIZE;

  Serial.print(rpm, 2);
  for (int i = 0; i < WINDOW_SIZE; i++) {
    Serial.print(',');
    Serial.print((accelBuffer[i] - mean) * G_PER_LSB, 4);
  }
  Serial.println();
}

void loop() {
  static unsigned long lastSampleUs = micros();
  unsigned long nowUs = micros();

  if (nowUs - lastSampleUs < SAMPLE_INTERVAL_US) {
    return;
  }
  lastSampleUs += SAMPLE_INTERVAL_US;

  int16_t raw = readRawX();
  accelBuffer[bufferIndex] = raw;
  bufferSum += raw;
  bufferIndex++;

  if (bufferIndex >= WINDOW_SIZE) {
    sendWindow();
    bufferIndex = 0;
    bufferSum = 0;
    // Transmission blocks sampling, so restart the sampling clock to
    // avoid a burst of catch-up samples at the start of the next window.
    lastSampleUs = micros();
  }
}