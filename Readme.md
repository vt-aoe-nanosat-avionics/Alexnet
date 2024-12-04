
# Setting Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install requirements.txt
```

# Compiling for Linux

## 1. Compile Tensorflow Lite

```bash
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile microlite
cd ..
```


## 2. Compoile Tensorflow Wrapper

```bash
mkdir build
g++ -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics -Wnon-virtual-dtor -Werror -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -Wsign-compare -Wdouble-promotion -Wunused-variable -Wunused-function -Wswitch -Wvla -Wall -Wextra -Wmissing-field-initializers -Wstrict-aliasing -Wno-unused-parameter -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_USE_CTIME -O2 -Itflite-micro/. -Itflite-micro/tensorflow/lite/micro/tools/make/downloads -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/kissfft -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/ruy -Itflite-micro/gen/linux_x86_64_default_gcc/genfiles/ -Itflite-micro/gen/linux_x86_64_default_gcc/genfiles/ -c target_x86/tflm_wrapper.cc -o build/tflm_wrapper.o
```


## 3. Compile main C code

```bash
g++ target_x86/alex.c target_x86/model_data.cc build/tflm_wrapper.o tflite-micro/gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a -o alex.out
```


# Compiling for Cortex M4


## 1. Compile Libopencm3

```bash
cd libopencm3
make
cd ..
```


## 2. Compile Tensorflow Lite

```bash
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_TOOLCHAIN_ROOT=/usr/bin/ microlite
cd ..
```


## 3. Compile Tensorflow Wrapper

```bash
arm-none-eabi-g++ -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -std=c++17 -Wall -Wextra -fno-threadsafe-statics -Itflite-micro/. -Itflite-micro/tensorflow/lite/micro/tools/make/downloads -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/kissfft -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/ruy -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/"ARMCM4"/Include -Itflite-micro/tensorflow/lite/micro -ltensorflow-microlite -c target_m4/tflm_wrapper.cc -o build/tflm_wrapper.o
```


## 4. Compile and link main C code

```bash
make
```


# Loading Models

The AlexNet model is loaded onto the off-chip flash on the board. The flash is only 16MB so any potential model to run on the MCU has to be smaller than 16MB. The model is loaded using a modified version of Tab Host. The modified version adds `cnn_write_model`, `cnn_write_cifar` and `app_infer`. The CNN functions are helper functions to assist in loading the model and changing the image data on the off-chip flash.

- `cnn_write_model`
    - **Description**: This function takes a C++ headerfile created using xxd and writes the byte array to the board's off-chip flash starting at 0x00001000. This function makes use of `common_erase` and `common_write` opcodes to write the model to the flash. The only arguement for this function is a relative path to the C++ file
    - **Arguements**: Relative path to header file
    - **Example**:
```
TAB> cnn_write_model target_x86/model_data.cc
```


- `cnn_write_cifar`
    - **Description**: This function takes an image from the CIFAR-10 testing dataset and writes the bytes of the image to the off-chip flash starting at 0x00000000. `cnn_write_cifar` erases tehe first sector and then reads an image from tensorflow and writes in to the flash. The only argument for this function is the index of the CIFAR-10 image to load to the flash. The image is supposed to display using matplotlib, but that caused strange bugs to appear. The only way to see the image is to use `python3 python/extract_cifar.py [IMAGE]` using the same index and the image will appear.
    - **Arguements**: Index of CIFAR-10 image
    - **Example**:
```
TAB> cnn_write_cifar 100
```

