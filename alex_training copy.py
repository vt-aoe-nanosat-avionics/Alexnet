import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
y_test = y_test.reshape(-1,)
y_train = y_train.reshape(-1,)

#x_train = x_train / 255.0
#x_test = x_test / 255.0

classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i])
plt.show() 

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


model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.optimizers.SGD(learning_rate=0.001),metrics=['accuracy'])
history=model.fit(x_train, y_train ,epochs=40,validation_data=(x_test, y_test))


fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val']) 
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])

#model.evaluate(x_test, y_test)

tf.keras.models.save_model(model, "models/model.keras")