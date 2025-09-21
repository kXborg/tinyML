#include <ArduinoBLE.h>
#include <Arduino_TensorFlowLite.h>
#include "digit_model.h"  // generated from xxd

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// BLE service/characteristics
BLEService digitService("19b10000-e8f2-537e-4f6c-d104768a1214");
BLECharacteristic imageChar("19b10001-e8f2-537e-4f6c-d104768a1214", BLEWriteWithoutResponse, 20);
BLEStringCharacteristic resultChar("19b10002-e8f2-537e-4f6c-d104768a1214", BLERead | BLENotify, 20);

uint8_t imageBuffer[64];
int bufferIndex = 0;

// TensorFlow Lite setup
const tflite::Model* model = tflite::GetModel(digit_model_tflite);
tflite::MicroAllOpsResolver resolver;
constexpr int tensorArenaSize = 8 * 1024;  // 8 KB
uint8_t tensorArena[tensorArenaSize];
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // BLE
  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1);
  }
  BLE.setLocalName("DigitNano");
  BLE.setAdvertisedService(digitService);
  digitService.addCharacteristic(imageChar);
  digitService.addCharacteristic(resultChar);
  BLE.addService(digitService);
  BLE.advertise();
  Serial.println("BLE advertising started...");

  // TensorFlow Lite
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensorArena, tensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void classifyDigit() {
  // Copy imageBuffer into input tensor (normalize 0-1)
  for (int i = 0; i < 64; i++) {
    input->data.f[i] = imageBuffer[i] / 255.0f;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Find predicted digit
  int maxIndex = 0;
  float maxVal = output->data.f[0];
  for (int i = 1; i < 10; i++) {
    if (output->data.f[i] > maxVal) {
      maxVal = output->data.f[i];
      maxIndex = i;
    }
  }

  Serial.print("Predicted digit: ");
  Serial.println(maxIndex);

  // Send result via BLE
  char resultStr[10];
  sprintf(resultStr, "%d", maxIndex);
  resultChar.writeValue(resultStr);
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      if (imageChar.written()) {
        int len = imageChar.valueLength();
        const uint8_t* data = imageChar.value();

        for (int i = 0; i < len; i++) {
          if (bufferIndex < 64) {
            imageBuffer[bufferIndex++] = data[i];
          }
        }

        if (bufferIndex >= 64) {
          bufferIndex = 0;
          classifyDigit();
        }
      }
    }
  }
}
