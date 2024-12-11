import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import cv2

(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
y_test = y_test.reshape(-1,)
y_train = y_train.reshape(-1,)


# Augment the training data
print("Augmenting training data...")

x_flipped = np.array([np.array(tf.image.flip_left_right(image)) for image in x_train.copy()])
#x_gray = np.array([np.array(tf.image.rgb_to_grayscale(image)) for image in x_train.copy()])
x_saturated = np.array([np.array(tf.image.adjust_saturation(image, 3)) for image in x_train.copy()])
x_brightness = np.array([np.array(tf.image.adjust_brightness(image, 0.5)) for image in x_train.copy()])
x_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_train.copy()])
x_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_train.copy()])
x_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_train.copy()])

#x_flipped_gray = np.array([np.array(tf.image.rgb_to_grayscale(image)) for image in x_flipped.copy()])
x_flipped_saturated = np.array([np.array(tf.image.adjust_saturation(image, 3)) for image in x_flipped.copy()])
x_flipped_brightness = np.array([np.array(tf.image.adjust_brightness(image, 0.5)) for image in x_flipped.copy()])
x_flipped_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_flipped.copy()])
x_flipped_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_flipped.copy()])
x_flipped_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_flipped.copy()])

#x_gray_brightness = np.array([np.array(tf.image.adjust_brightness(image, 0.5)) for image in x_gray.copy()])
#x_gray_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_gray.copy()])
#x_gray_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_gray.copy()])
#x_gray_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_gray.copy()])

x_saturated_brightness = np.array([np.array(tf.image.adjust_brightness(image, 0.5)) for image in x_saturated.copy()])
x_saturated_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_saturated.copy()])
x_saturated_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_saturated.copy()])
x_saturated_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_saturated.copy()])

x_brightness_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_brightness.copy()])
x_brightness_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_brightness.copy()])
x_brightness_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_brightness.copy()])

#x_flipped_gray_brightness = np.array([np.array(tf.image.adjust_brightness(image, 0.5)) for image in x_flipped_gray.copy()])
#x_flipped_gray_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_flipped_gray.copy()])
#x_flipped_gray_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_flipped_gray.copy()])
#x_flipped_gray_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_flipped_gray.copy()])

x_flipped_saturated_brightness = np.array([np.array(tf.image.adjust_brightness(image, 0.5)) for image in x_flipped_saturated.copy()])
x_flipped_saturated_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_flipped_saturated.copy()])
x_flipped_saturated_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_flipped_saturated.copy()])
x_flipped_saturated_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_flipped_saturated.copy()])

x_flipped_brightness_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_flipped_brightness.copy()])
x_flipped_brightness_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_flipped_brightness.copy()])
x_flipped_brightness_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_flipped_brightness.copy()])

#x_gray_brightness_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_gray_brightness.copy()])
#x_gray_brightness_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_gray_brightness.copy()])
#x_gray_brightness_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_gray_brightness.copy()])

x_saturated_brightness_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_saturated_brightness.copy()])
x_saturated_brightness_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_saturated_brightness.copy()])
x_saturated_brightness_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_saturated_brightness.copy()])

#x_flipped_gray_brightness_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_flipped_gray_brightness.copy()])
#x_flipped_gray_brightness_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_flipped_gray_brightness.copy()])
#x_flipped_gray_brightness_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_flipped_gray_brightness.copy()])

x_flipped_saturated_brightness_90deg = np.array([np.array(tf.image.rot90(image)) for image in x_flipped_saturated_brightness.copy()])
x_flipped_saturated_brightness_180deg = np.array([np.array(tf.image.rot90(image, k=2)) for image in x_flipped_saturated_brightness.copy()])
x_flipped_saturated_brightness_270deg = np.array([np.array(tf.image.rot90(image, k=3)) for image in x_flipped_saturated_brightness.copy()])

x_train = np.concatenate((x_train, x_flipped, x_saturated, x_brightness, x_90deg, x_180deg, x_270deg, x_flipped_saturated, x_flipped_brightness, x_flipped_90deg, x_flipped_180deg, x_flipped_270deg, x_saturated_brightness, x_saturated_90deg, x_saturated_180deg, x_saturated_270deg, x_brightness_90deg, x_brightness_180deg, x_brightness_270deg, x_flipped_saturated_brightness, x_flipped_saturated_90deg, x_flipped_saturated_180deg, x_flipped_saturated_270deg, x_flipped_brightness_90deg, x_flipped_brightness_180deg, x_flipped_brightness_270deg, x_saturated_brightness_90deg, x_saturated_brightness_180deg, x_saturated_brightness_270deg, x_flipped_saturated_brightness_90deg, x_flipped_saturated_brightness_180deg, x_flipped_saturated_brightness_270deg))
y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train))


classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# for i in range(0,9):
#     plt.subplot(330+1+i)
#     plt.imshow(x_train[i])
# plt.show() 

print(x_train.shape)

model = models.Sequential()
model.add(layers.Resizing(64,64, interpolation="bilinear", input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(64,64,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
model.add(layers.MaxPool2D(pool_size=(3,3)))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
model.add(layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
model.add(layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='SGD', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.summary()


model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.optimizers.SGD(learning_rate=0.003),metrics=['accuracy'])
history=model.fit(x_train, y_train ,epochs=50, batch_size=1024, validation_data=(x_test, y_test))


fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val']) 
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])

model.evaluate(x_test, y_test)

tf.keras.models.save_model(model, "models/model.keras")