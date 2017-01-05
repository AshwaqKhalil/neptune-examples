#
#  Copyright (c) 2017, deepsense.io
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
from deepsense import neptune


def _evaluate_accuracy(submission, true_labels):
    return np.mean(np.argmax(submission, axis=1) == true_labels)


def _evaluate_cross_entropy(submission, true_labels):
    return -np.mean(np.log(np.choose(true_labels, submission.T)))


def _prepare_neptune(neptune_context, tags=None):
    neptune_context.job.tags.extend(tags or [])

    accuracy_channel = neptune_context.job.create_channel(
        "accuracy",
        neptune.ChannelType.NUMERIC)
    cross_entropy_channel = neptune_context.job.create_channel(
        "cross entropy",
        neptune.ChannelType.NUMERIC)

    return accuracy_channel, cross_entropy_channel


def _send_to_neptune(channels, values):
    for (channel, value) in zip(channels, values):
        channel.send(1, value)


def evaluate_and_send_to_neptune(submission, true_labels, neptune_context, tags=None):
    # evaluating
    metric_values = (_evaluate_accuracy(submission, true_labels), _evaluate_cross_entropy(submission, true_labels))

    # preparing Neptune
    metric_channels = _prepare_neptune(neptune_context, tags)

    # sending to Neptune
    _send_to_neptune(metric_channels, metric_values)
