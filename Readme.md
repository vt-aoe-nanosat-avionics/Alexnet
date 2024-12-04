
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

## Linux

A python script is provided to extract a CIFAR-10 image from the tensorflow dataset. This script extracts the image data and writes it bytewise to a file called cifarData. This file is read by the main c code and is used for the AlexNet inference.The script uses matplotlib to display the selected image
```
python3 python/extract_cifar.py 1001
```


## Cortex M4

The only method to load a model onto the off-chip flash for this demo version of AlexNet is to use different firmware and write the modal manually. This process is very specific, so the steps are below to ensure the model is loaded correctly.

1. Erase off-chip flash
Before the model can be written to the off-chip flash, it first needs to be erased of any previous data. The erase needs to start at 0x00001000 and be for enough sectors to hold the entire model. It should be 1150 sectors, but calclating this value may be necessary for modified models.

2. Load the model
The model is loaded starting at 0x00001004. This address is chosen because it is far enough from the start of the off-chip flash to allow erasing image data without effecting the model, while also giving the most space for the model. The address is 0x00001004 because the first 4 bytes starting at 0x00001000 when converted to a 32 bit integer contain the size of the model in bytes.

The same process can be followed for loading the image data, but starting at 0x00000000 instead. Only the first sector should be erased for the image data. The same process as above can be used to extract the image data from the CIFAR-10 dataset, and then the cifarData file can be written to 0x00000004

**If the model is corrupted, the MCU will hang when it trys to initialize the model at startup**

