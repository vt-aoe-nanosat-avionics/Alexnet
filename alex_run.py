import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorflow.keras import datasets

image = int(sys.argv[1])


(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()

y_test = y_test.reshape(-1,)
y_train = y_train.reshape(-1,)


model = tf.keras.models.load_model("models/model.keras")

predictions = model.predict(x_test)


interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.float32)
for i in range(32):
    for j in range(32):
        for k in range(3):
            input_data[0][i][j][k][0] = x_test[image][i][j][k]
            print(int(input_data[0][i][j][k][0]), end='\t')
    print('\n', end='')
print('\n')
print(predictions[image])
for i in range(32):
    for j in range(32):
        print(int(x_test[image][i][j]), end='\t')
        test = 0
    print('\n', end='')

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = x_test[0].numpy()
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

print(y_test[image])