from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.callbacks import Callback, TensorBoard
from PIL import Image

from deepsense import neptune

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

ctx = neptune.Context()
ctx.integrate_with_tensorflow()
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)



false_predictions_channel = ctx.job.create_channel(
    name='false_predictions',
    channel_type=neptune.ChannelType.IMAGE)


# Prepare an image of an incorrectly recognized digit to be sent to Neptune.
def false_prediction_neptune_image(image_2d_float, index, epoch_number, prediction, actual):
    false_prediction_image = Image.fromarray((255 * image_2d_float).astype('uint8'))
    image_name = '(epoch {}) #{}'.format(epoch_number, index)
    image_description = 'Predicted: {}, actual: {}.'.format(prediction, actual)
    return neptune.Image(
        name=image_name,
        description=image_description,
        data=false_prediction_image)

class EpochEndCallback(Callback):
    def __init__(self):
        self.epoch_id = 0
        self.false_predictions = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # Predict the digits for images of the test set.
        validation_predictions = model.predict_classes(X_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        for index, (prediction, actual) in enumerate(zip(validation_predictions, Y_test.argmax(axis=1))):
            if prediction != actual:
                self.false_predictions += 1
                false_prediction_image = false_prediction_neptune_image(
                    X_test[index,:,:,0], index, self.epoch_id, prediction, actual)
                false_predictions_channel.send(x=self.false_predictions, y=false_prediction_image)



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
          callbacks=[tbCallBack, EpochEndCallback()])
