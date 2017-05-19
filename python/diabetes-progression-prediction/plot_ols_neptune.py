#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the mean squared error
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the mean squared error and the variance score are also
calculated.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

# All contributions by deepsense.io:
# Copyright (c) 2016, deepsense.io
# All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\
#
import matplotlib
# Agg is used to generate images without having a window appear
# The order of matplotlib imports have to be like this
# Reason: https://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
matplotlib.use('Agg')
import numpy as np
from sklearn import datasets, linear_model

# The additional libraries.
import io
import time
from deepsense import neptune
from PIL import Image
from matplotlib import pyplot as plt

ctx = neptune.Context()

# A channel to send the Mean Squared Error metric value.
mse_channel = ctx.job.create_channel(
    name='MSE',
    channel_type=neptune.ChannelType.NUMERIC)

# A channel to send the regression chart.
regression_chart_channel = ctx.job.create_channel(
    name='Regression chart',
    channel_type=neptune.ChannelType.IMAGE)

# A channel to log information about job's execution.
logs_channel = ctx.job.create_channel(
    name='logs',
    channel_type=neptune.ChannelType.TEXT)

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Add a tag containing the name of the feature.
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
used_feature_name = feature_names[ctx.params.feature_index]
ctx.job.tags.append('diabetes-feature-' + used_feature_name)

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, ctx.params.feature_index]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
logs_channel.send(x=1, y='Coefficients: ' + str(regr.coef_))

# The mean square error
mse = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
mse_channel.send(x=time.time(), y=mse)
logs_channel.send(x=2, y="Mean squared error: %.2f" % mse)

# Explained variance score: 1 is perfect prediction
logs_channel.send(x=3, y='Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

# Convert the chart to an image.
image_buffer = io.BytesIO()
plt.savefig(image_buffer, format='png')
image_buffer.seek(0)

# Send the chart to Neptune through an image channel.
regression_chart_channel.send(
    x=time.time(),
    y=neptune.Image(
        name='Regression chart',
        description='A chart containing predictions and target values '
                    'for diabetes progression regression. '
                    'Feature used: ' + used_feature_name,
        data=Image.open(image_buffer)))
