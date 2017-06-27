from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.callbacks import TensorBoard

from deepsense import neptune

# the easiest integration
ctx = neptune.Context()
ctx.integrate_with_tensorflow()
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# wget http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat
# load data
data = io.loadmat("notMNIST_small.mat")
X = data['images']
y = data['labels']

resolution = 28  # 28x28 images, grayscale
classes = 10     # 10 letters: ABCDEFGHIJ

# transforming data for TensorFlow backend
X = np.transpose(X, (2, 0, 1))
X = X.reshape((-1, resolution, resolution, 1))
X = X.astype('float32') / 255.

# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
y = y.astype('int32')
Y = np_utils.to_categorical(y, classes)

# splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    random_state=137)

# create neural network architeture

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


model.fit(X_train, Y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, Y_test),
          verbose=2,
          callbacks=[tbCallBack])
