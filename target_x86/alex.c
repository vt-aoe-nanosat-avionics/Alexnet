#include <stdio.h>
#include <stdint.h>

#include "tflm_wrapper.h"
#include "model_data.h"


void run_alex_cnn(void);

int main(void) {
    run_alex_cnn();
    return 0;
}

void run_alex_cnn(void) {

    tflm_init(alexnet_model_tflite);

    float* input = tflm_get_input_buffer(0);

    uint8_t input_data[32*32*3];
    int len;
    FILE *fp = fopen("cifarData", "rb");
    if(!fp) {
        printf("Error opening file\n");
        return;
    }
    len = fread(input_data, 1, 32*32*3, fp);
    fclose(fp);


    // Populate the input buffer with test data.
    for (int i = 0; i < 32 * 32 * 3; i++) {
        input[i] = input_data[i]; // Replace with actual image data.
    }

    // Run inference.
    tflm_invoke();

    // Retrieve output predictions.
    const float* output = tflm_get_output_buffer(0);

    //printf("Output:\n");
    printf("airplane\tautomobile\tbird\t\tcat\t\tdeer\t\tdog\t\tfrog\t\thorse\t\tship\t\ttruck\n");
    printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t\n", output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]);

}
