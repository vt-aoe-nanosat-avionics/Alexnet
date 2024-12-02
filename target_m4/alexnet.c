#include <stdio.h>
#include <stdint.h>

#include <libopencm3/stm32/usart.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/quadspi.h>
#include "ta-expt/IS25LP128F.h"

#include "tflm_wrapper.h"
#include "alexnet.h"

void init_alexnet_cnn(void) {
    quadspi_wait_while_busy();
    uint32_t ccr = 0;
    ccr = quadspi_prepare_funcion_mode(ccr, QUADSPI_CCR_FMODE_MEMMAP);
    ccr = quadspi_prepare_address_mode(ccr, QUADSPI_CCR_MODE_4LINE);
    ccr = quadspi_prepare_address_size(ccr, QUADSPI_CCR_SIZE_32BIT);
    ccr = quadspi_prepare_data_mode(ccr, QUADSPI_CCR_MODE_4LINE);
    ccr = quadspi_prepare_dummy_cycles(ccr, 6);
    ccr = quadspi_prepare_instruction_mode(ccr, QUADSPI_CCR_MODE_4LINE);
    ccr = quadspi_prepare_instruction(ccr, IS25LP128F_CMD_FAST_READ);
    ccr = quadspi_prepare_alternative_bytes_mode(ccr, QUADSPI_CCR_MODE_NONE);
    quadspi_write_ccr(ccr);
    
    uint8_t* data = (uint8_t*)0x90001000;
    unsigned int alexnet_model_tflite_len = (data[0]<<24)|(data[1]<<16)|(data[2]<<8)|(data[3]<<0);
    const unsigned char* alexnet_model_tflite = (unsigned char*)0x90001004;
    tflm_init(alexnet_model_tflite);

    ccr = quadspi_prepare_funcion_mode(ccr, QUADSPI_CCR_FMODE_IREAD);
    quadspi_write_ccr(ccr);
}

float* run_alexnet_cnn(void) {
    uint32_t ccr = 0;
    ccr = quadspi_prepare_funcion_mode(ccr, QUADSPI_CCR_FMODE_MEMMAP);
    ccr = quadspi_prepare_address_mode(ccr, QUADSPI_CCR_MODE_4LINE);
    ccr = quadspi_prepare_address_size(ccr, QUADSPI_CCR_SIZE_32BIT);
    ccr = quadspi_prepare_data_mode(ccr, QUADSPI_CCR_MODE_4LINE);
    ccr = quadspi_prepare_dummy_cycles(ccr, 6);
    ccr = quadspi_prepare_instruction_mode(ccr, QUADSPI_CCR_MODE_4LINE);
    ccr = quadspi_prepare_instruction(ccr, IS25LP128F_CMD_FAST_READ);
    ccr = quadspi_prepare_alternative_bytes_mode(ccr, QUADSPI_CCR_MODE_NONE);
    quadspi_write_ccr(ccr);
    uint8_t* input_data = (uint8_t*)0x90000004;

    float* input = tflm_get_input_buffer(0);
    
    if(input == NULL)
    {
        return NULL;
    }

    // Populate the input buffer with test data.
    for (int i = 0; i < 32*32*3; i++) {
        input[i] = input_data[i];
    }

    // Run inference.
    tflm_invoke();
    
    float* output = tflm_get_output_buffer(0);
    
    ccr = quadspi_prepare_funcion_mode(ccr, QUADSPI_CCR_FMODE_IREAD);
    quadspi_write_ccr(ccr);

    return output;
}
