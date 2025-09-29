#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "digits_model_quant_19k.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define IMG_SIZE 28*28

uint8_t image_buffer[IMG_SIZE];

// TensorFlow Lite globals
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int tensorArenaSize = 40 * 1024;  // adjust depending on model
uint8_t tensorArena[tensorArenaSize];

extern "C" void DebugLog(const char* s) {
  Serial.print(s);  // send TFLite logs to Serial
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  model = tflite::GetModel(digits_model_quant_19k);
  interpreter = new tflite::MicroInterpreter(model, resolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Ready for image");
}

void loop() {
  // Check if enough bytes have arrived
  if (Serial.available() >= IMG_SIZE) {
    // Read image bytes into buffer
    Serial.readBytes((char*)image_buffer, IMG_SIZE);

    // Copy to TFLite input tensor
    for (int i = 0; i < IMG_SIZE; i++) {
      input->data.uint8[i] = image_buffer[i];
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Inference failed!");
      return;
    }

    // Find predicted class
    int best = 0;
    for (int i = 1; i < output->dims->data[1]; i++) {
      if (output->data.uint8[i] > output->data.uint8[best]) {
        best = i;
      }
    }

    // Send prediction back over Serial
    Serial.println(best);  // newline required for Python readLine()

    // Optional: signal ready for next image
    Serial.println("READY");
  }
}

