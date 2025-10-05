#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "digits_model_28x28_int8.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define IMG_SIZE 28 * 28  // 784 bytes

uint8_t image_buffer[IMG_SIZE];
int received_bytes = 0;
bool image_ready = false;

// TensorFlow Lite globals
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int tensorArenaSize = 30 * 1024;
uint8_t tensorArena[tensorArenaSize];

// BLE configuration
BLEService digitService("19b10000-e8f2-537e-4f6c-d104768a1214");  // custom service UUID
BLECharacteristic imageChar("19b10001-e8f2-537e-4f6c-d104768a1214", BLEWriteWithoutResponse | BLEWrite, IMG_SIZE);
BLECharacteristic resultChar("19b10002-e8f2-537e-4f6c-d104768a1214", BLERead | BLENotify, 20);

extern "C" void DebugLog(const char* s) {
  Serial.print(s);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Starting BLE Digit Classifier...");

  Serial.println("Initializing BLE...");
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }
  Serial.println("BLE initialized.");

  BLE.setLocalName("DigitClassifier");
  BLE.setAdvertisedService(digitService);

  digitService.addCharacteristic(imageChar);
  digitService.addCharacteristic(resultChar);

  BLE.addService(digitService);
  imageChar.writeValue((uint8_t)0);
  resultChar.writeValue("Waiting");

  Serial.println("Starting BLE advertise...");
  BLE.advertise();
  Serial.println("BLE Device Active, Waiting for Connection...");

  Serial.println("Initializing TensorFlow Lite...");
  model = tflite::GetModel(digits_model_28x28_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  interpreter = new tflite::MicroInterpreter(model, resolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  Serial.println("Allocating tensors...");
  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete. Ready to receive images over BLE.");
}


void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    received_bytes = 0;
    image_ready = false;

    while (central.connected()) {
      if (imageChar.written()) {
        int len = imageChar.valueLength();
        const uint8_t* data = imageChar.value();

        for (int i = 0; i < len && received_bytes < IMG_SIZE; i++) {
          image_buffer[received_bytes++] = data[i];
        }

        if (received_bytes >= IMG_SIZE) {
          image_ready = true;
          received_bytes = 0;
        }
      }

      if (image_ready) {
        Serial.println("Image received. Running inference...");
        runInference();
        image_ready = false;
      }
    }

    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}

void runInference() {
  // Map received 0..255 -> int8 -128..127
  for (int i = 0; i < IMG_SIZE; i++) {
      input->data.int8[i] = static_cast<int8_t>(image_buffer[i] - 128);
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
      Serial.println("Inference failed!");
      resultChar.writeValue("Error");
      return;
  }

  int best = 0;
  for (int i = 1; i < output->dims->data[1]; i++) {
      if (output->data.int8[i] > output->data.int8[best]) best = i;
  }

  char result[16];
  sprintf(result, "Digit:%d", best);
  resultChar.writeValue(result);
  Serial.print("Predicted: ");
  Serial.println(result);
}

