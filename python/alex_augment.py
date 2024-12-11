import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorflow.keras import datasets
import cv2
import random



(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()

y_test = y_test.reshape(-1,)
y_train = y_train.reshape(-1,)

image = random.randint(0, len(x_test))
factor = 100
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cv2.namedWindow(labels[y_test[image]], cv2.WINDOW_NORMAL)
cv2.imshow(labels[y_test[image]], cv2.resize(x_test[image], (32*factor, 32*factor), interpolation= cv2.INTER_NEAREST))


x_flipped = np.array(tf.image.flip_left_right(x_test[image].copy()))

cv2.namedWindow(labels[y_test[image]] + ' flipped', cv2.WINDOW_NORMAL)
cv2.imshow(labels[y_test[image]] + ' flipped', cv2.resize(x_flipped, (32*factor, 32*factor), interpolation= cv2.INTER_NEAREST))


x_gray = np.array(tf.image.rgb_to_grayscale(x_test[image].copy()))

cv2.namedWindow(labels[y_test[image]] + ' gray', cv2.WINDOW_NORMAL)
cv2.imshow(labels[y_test[image]] + ' gray', cv2.resize(x_gray, (16*factor, 16*factor), interpolation= cv2.INTER_NEAREST))

x_flipped_gray = np.array(tf.image.rgb_to_grayscale(x_flipped.copy()))

cv2.namedWindow(labels[y_test[image]] + ' flipped gray', cv2.WINDOW_NORMAL)
cv2.imshow(labels[y_test[image]] + ' flipped gray', cv2.resize(x_flipped_gray, (16*factor, 16*factor), interpolation= cv2.INTER_NEAREST))

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

