import pandas as pd
from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# pip install git+git://github.com/stared/keras-sequential-ascii.git
from keras_sequential_ascii import sequential_model_to_ascii_printout

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout


# XXX data loading in a different file?
# wget http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat
# load data
data = io.loadmat("notMNIST_small.mat")

# transform data
X = data['images']
y = data['labels']

# 28x28 images, grayscale
resolution = 28
# 10 letters: ABCDEFGHIJ
classes = 10

X = np.transpose(X, (2, 0, 1))

y = y.astype('int32')
X = X.astype('float32') / 255.

# channel for X
X = X.reshape((-1, resolution, resolution, 1))

# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
Y = np_utils.to_categorical(y, 10)

# splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    random_state=137)

# create network

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu',
                 input_shape=(resolution, resolution, 1)))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# visualize network architecture
sequential_model_to_ascii_printout(model)

model.fit(X_train, Y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, Y_test), callbacks=[])
