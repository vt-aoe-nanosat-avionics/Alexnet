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

#evaluaton = model.evaluate(x_test, y_test)
predictions = model.predict(x_test[image].reshape(1,32,32,3))
print(predictions)


interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.zeros(input_details[0]['shape'], dtype=np.float32)
for i in range(32):
    for j in range(32):
        for k in range(3):
            input_data[0][i][j][k] = x_test[image][i][j][k]

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = x_test[0].numpy()
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

print(y_test[image])