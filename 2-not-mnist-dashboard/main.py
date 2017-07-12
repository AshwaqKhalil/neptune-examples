from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.callbacks import Callback, TensorBoard
# from PIL import Image

# from deepsense import neptune
import neptuner

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

# Neptune integration

false_predictions_channel = neptuner.image_channel('false_predictions')
letters = "ABCDEFGHIJ"

class EpochEndCallback(Callback):
    def __init__(self, images_per_epoch=-1):
        self.epoch_id = 0
        self.images_per_epoch = images_per_epoch

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # Predict the digits for images of the test set.
        validation_predictions = model.predict_classes(X_test)
        scores = model.predict(X_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        image_per_epoch = 0
        for index, (prediction, actual) in enumerate(zip(validation_predictions, Y_test.argmax(axis=1))):
            if prediction != actual:
                if image_per_epoch == self.images_per_epoch:
                    break
                image_per_epoch += 1

                false_predictions_channel.send_array(
                    X_test[index,:,:,0],
                    name="[{epoch}] pred: {pred} true: {true}".format(epoch=self.epoch_id,
                                                          pred=letters[prediction],
                                                          true=letters[actual]),
                    description="\n".join(["{} {:5.1f}% {}".format(letters[i], 100 * score, "!!!" if i == actual else "")
                                           for i, score in enumerate(scores[index])])
                )


# create neural network architeture

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu',
                 input_shape=(resolution, resolution, 1)))
# model.add(Conv2D(16, (3, 3), activation='relu'))  # uncomment!
model.add(MaxPool2D())

# model.add(Conv2D(32, (3, 3), activation='relu'))  # uncomment!
# model.add(Conv2D(32, (3, 3), activation='relu'))  # uncomment!
model.add(MaxPool2D())

model.add(Flatten())
# model.add(Dropout(0.5))  # uncomment!
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, Y_test),
          verbose=2,
          callbacks=[EpochEndCallback(images_per_epoch=20)])
