# -*- coding: utf-8 -*-
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015, Google, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015, the respective contributors.
# All rights reserved.
#
# All contributions by deepsense.io:
# Copyright (c) 2016, deepsense.io
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Original source code with the complete history of contributions can be found:
# https://github.com/fchollet/keras/blob/cc92025fdc862e00cf787cc309c741e8944ed0a7/examples/mnist_cnn.py
#

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import zip
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from keras.callbacks import Callback
from deepsense import neptune
from PIL import Image
import time

ctx = neptune.Context()

# Channels to send metrics' values (loss and accuracy) at the end of every batch.
batch_train_loss_channel = ctx.job.create_channel(
    name='batch_train_loss',
    channel_type=neptune.ChannelType.NUMERIC)
batch_train_acc_channel = ctx.job.create_channel(
    name='batch_train_acc',
    channel_type=neptune.ChannelType.NUMERIC)

# Channels to send metrics' values (loss and accuracy) at the end of every epoch.
epoch_train_loss_channel = ctx.job.create_channel(
    name='epoch_train_loss',
    channel_type=neptune.ChannelType.NUMERIC)
epoch_train_acc_channel = ctx.job.create_channel(
    name='epoch_train_acc',
    channel_type=neptune.ChannelType.NUMERIC)

epoch_validation_loss_channel = ctx.job.create_channel(
    name='epoch_validation_loss',
    channel_type=neptune.ChannelType.NUMERIC)
epoch_validation_acc_channel = ctx.job.create_channel(
    name='epoch_validation_acc',
    channel_type=neptune.ChannelType.NUMERIC)

# A channel to send info about the progress of the job.
logging_channel = ctx.job.create_channel(
    name='logging_channel',
    channel_type=neptune.ChannelType.TEXT)

# A channel to send images of digits that were not recognized correctly.
false_predictions_channel = ctx.job.create_channel(
    name='false_predictions',
    channel_type=neptune.ChannelType.IMAGE)

# Charts displaying training metrics' values updated at the end of every batch.
ctx.job.create_chart(
    name='Batch training loss',
    series={
        'training loss': batch_train_loss_channel
    }
)

ctx.job.create_chart(
    name='Batch training accuracy',
    series={
        'training': batch_train_acc_channel
    }
)

# Charts displaying training and validation metrics updated at the end of every epoch.
ctx.job.create_chart(
    name='Epoch training and validation loss',
    series={
        'training': epoch_train_loss_channel,
        'validation': epoch_validation_loss_channel
    }
)

ctx.job.create_chart(
    name='Epoch training and validation accuracy',
    series={
        'training': epoch_train_acc_channel,
        'validation': epoch_validation_acc_channel
    }
)


# Format the timestamp in a human-readable format.
def format_timestamp(timestamp):
    return time.strftime('%H:%M:%S', time.localtime(timestamp))


# Prepare an image of an incorrectly recognized digit to be sent to Neptune.
def false_prediction_neptune_image(raw_image, index, epoch_number, prediction, actual):
    false_prediction_image = Image.fromarray(raw_image)
    image_name = '(epoch {}) #{}'.format(epoch_number, index)
    image_description = 'Predicted: {}, actual: {}.'.format(prediction, actual)
    return neptune.Image(
        name=image_name,
        description=image_description,
        data=false_prediction_image)


class BatchEndCallback(Callback):
    def __init__(self):
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1

        # Send training metrics.
        batch_train_loss_channel.send(x=self.batch_id, y=float(logs.get('loss')))
        batch_train_acc_channel.send(x=self.batch_id, y=float(logs.get('acc')))

        # Log the end of the batch.
        timestamp = time.time()
        batch_end_message = '{} Batch {} finished, batch size = {}.'.format(
            format_timestamp(timestamp), self.batch_id, logs.get('size'))

        logging_channel.send(x=timestamp, y=batch_end_message)


class EpochEndCallback(Callback):
    def __init__(self):
        self.epoch_id = 0
        self.false_predictions = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # Send training and validation metrics.
        epoch_train_loss_channel.send(x=self.epoch_id, y=float(logs.get('loss')))
        epoch_train_acc_channel.send(x=self.epoch_id, y=float(logs.get('acc')))

        epoch_validation_loss_channel.send(x=self.epoch_id, y=float(logs.get('val_loss')))
        epoch_validation_acc_channel.send(x=self.epoch_id, y=float(logs.get('val_acc')))

        # Predict the digits for images of the test set.
        validation_predictions = model.predict_classes(X_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        for index, (prediction, actual) in enumerate(zip(validation_predictions, y_test)):
            if prediction != actual:
                self.false_predictions += 1
                false_prediction_image = false_prediction_neptune_image(
                    raw_X_test[index], index, self.epoch_id, prediction, actual)
                false_predictions_channel.send(x=self.false_predictions, y=false_prediction_image)

        # Log the end of the epoch.
        timestamp = time.time()
        epoch_end_message = '{} Epoch {}/{} finished.'.format(
            format_timestamp(timestamp), self.epoch_id, nb_epoch)

        logging_channel.send(x=timestamp, y=epoch_end_message)


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (ctx.params.kernel_size, ctx.params.kernel_size)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# let's store unprocessed Xs for the later use:
raw_X_test = X_test

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test),
          callbacks=[BatchEndCallback(), EpochEndCallback()])

accuracy = model.evaluate(X_test, Y_test, verbose=0)[1]

timestamp = time.time()
logging_channel.send(
    x=timestamp,
    y='{} Evaluation finished. Overall accuracy = {}.'.format(
        format_timestamp(timestamp), accuracy))
