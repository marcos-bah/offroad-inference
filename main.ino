/***************************************************
  This is Consentium's TinyML library
  ----> https://docs.consentiuminc.online/
  Check out the links above for our tutorials and product diagrams.

  This Consentium's TinyML library works only for ESP32/Raspberry Pi Pico W compatible Edge boards. 
  
  Written by Debjyoti Chowdhury for Consentium.
  MIT license, all text above must be included in any redistribution
 ****************************************************/

// Uncomment library definition, according to your board version


#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
//#include <Wire.h>            // I2C library for ESP32
#include <EdgeNeuron.h>      // TensorFlow Lite wrapper for Arduino
#include "model.h"           // Trained model

Adafruit_MPU6050 mpu;
TwoWire CustomI2C0(i2c0, 0, 1); // i2c0 é a instância do controlador I²C padrão

const float accelerationThreshold = 2.5;  // Threshold (in G values) to detect a "gesture" start
const int numSamples = 50;                // Number of samples for a single gesture
int samplesRead = 0;                           // Sample counter
const int inputLength = 300;               // Input tensor size (6 values * 119 samples)

// Tensor Arena memory area for TensorFlow Lite to store tensors
constexpr int tensorArenaSize = 8 * 1024;
alignas(16) byte tensorArena[tensorArenaSize];

// Gesture labels table
const char* GESTURES[] = {
  "fall",
  "movement",
  "idle"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

const int ledIdle = 20;
const int ledMovement = 19;
const int ledFall = 18;

void initButtonLed() {
  pinMode(ledIdle, OUTPUT);
  pinMode(ledMovement, OUTPUT);
  pinMode(ledFall, OUTPUT);

  pinMode(LED_BUILTIN, OUTPUT);
}

void setup() {
  Serial.begin(115200);
  
  //Wire.begin();  // Start I2C communication

  // Initialize MPU6050 sensor
  // Try to initialize! 0x68 addrs, custom i2c, sensor_id let it be 0(zero)
  if (!mpu.begin(0x68,&CustomI2C0,0)) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }

  Serial.println("MPU6050 initialized.");

  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("");
  delay(100);
  
  // Set accelerometer and gyroscope sampling rates
  //mpu.calcOffsets();  // Calibrate MPU6050
  
  Serial.println();
  Serial.println("Initializing TensorFlow Lite model...");
  if (!initializeModel(model, tensorArena, tensorArenaSize)) {
    Serial.println("Model initialization failed!");
    while (true);  // Stop execution on failure
  }
  Serial.println("Model initialization done.");
  initButtonLed();
}

void lightUpLed(String gesture) {
  // Apague todos os LEDs primeiro
  digitalWrite(ledIdle, LOW);
  digitalWrite(ledMovement, LOW);
  digitalWrite(ledFall, LOW);

  // Acenda o LED correspondente ao gesto
  if (gesture == "idle") {
    digitalWrite(ledIdle, HIGH);
  } else if (gesture == "movement") {
    digitalWrite(ledMovement, HIGH);
  } else if (gesture == "fall") {
    digitalWrite(ledFall, HIGH);
  }
}

void loop() {
  samplesRead = 0;

  // Collect data for gesture
  while (samplesRead < numSamples) {
    digitalWrite(LED_BUILTIN, HIGH);
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Read accelerometer and gyroscope values
    float aX = a.acceleration.x;
    float aY = a.acceleration.y;
    float aZ = a.acceleration.z;
    float gX = g.gyro.x;
    float gY = g.gyro.y;
    float gZ = g.gyro.z;

    // Normalize sensor data (since the model was trained on normalized values)
    // aX = (aX + 4.0) / 8.0;
    // aY = (aY + 4.0) / 8.0;
    // aZ = (aZ + 4.0) / 8.0;
    // gX = (gX + 2000.0) / 4000.0;
    // gY = (gY + 2000.0) / 4000.0;
    // gZ = (gZ + 2000.0) / 4000.0;

    // Place the 6 values (acceleration and gyroscope) into the model's input tensor
    setModelInput(aX, samplesRead * 6 + 0);
    setModelInput(aY, samplesRead * 6 + 1);
    setModelInput(aZ, samplesRead * 6 + 2);
    setModelInput(gX, samplesRead * 6 + 3);
    setModelInput(gY, samplesRead * 6 + 4);
    setModelInput(gZ, samplesRead * 6 + 5);

    samplesRead++;

    delay(80);

    // Once all samples are collected, run the inference
    if (samplesRead == numSamples) {
      digitalWrite(LED_BUILTIN, LOW);
      if (!runModelInference()) {
        Serial.println("Inference failed!");
        return;
      }

      int highestIndex = 0;
      float highestValue = 0.0;

      // Retrieve output values and print them
      for (int i = 0; i < NUM_GESTURES; i++) {
        float value = getModelOutput(i) * 100;
        if (value > highestValue) {
          highestValue = value;
          highestIndex = i;
        }

        Serial.print(GESTURES[i]);
        Serial.print(": ");
        Serial.print(value, 2);
        Serial.println("%");
      }

      String gestureWithHighestValue = GESTURES[highestIndex];
      Serial.print("Maior gesto: ");
      Serial.println(gestureWithHighestValue);
      lightUpLed(gestureWithHighestValue);

      Serial.println();
    }
  }
}

