from tensorflow.keras import datasets
from matplotlib import pyplot as plt
import tensorflow as tf
import sys

file = int(sys.argv[1])

(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()

y_test = y_test.reshape(-1,)
y_train = y_train.reshape(-1,)

x_train = x_train
x_test = x_test

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data = []


for i in range(32):
  for j in range(32):
      for k in range(3):
        data.append(int(x_test[file][i][j][k]))

with open('cifarData', 'wb') as f:
  f.write(bytearray(data))

plt.imshow(x_test[file])
plt.title(labels[y_test[file]])
plt.show()