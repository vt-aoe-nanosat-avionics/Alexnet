#include <tensorflow/lite/micro/kernels/micro_ops.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "tflm_wrapper.h"


namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 100000 * 1024; //  100 KB
  __attribute__((aligned(16))) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


extern "C" void tflm_init(const uint8_t* model_data) {
    model = ::tflite::GetModel(model_data);

    static tflite::MicroMutableOpResolver<9> micro_op_resolver;

    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddResizeBilinear();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
}

extern "C" float* tflm_get_input_buffer(int index) {
    if (!interpreter) return nullptr;
    input = interpreter->input(index);
    return input ? input->data.f : nullptr;
}

extern "C" const float* tflm_get_output_buffer(int index) {
    if (!interpreter) return nullptr;
    output = interpreter->output(index);
    return output ? output->data.f : nullptr;
}

extern "C" void tflm_invoke() {
    if (interpreter) {
        interpreter->Invoke();
    }
}