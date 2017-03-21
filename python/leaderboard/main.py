#
# Copyright (c) 2017, deepsense.io
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf

import cifar10_train

import cifar10_submission
import evaluation

from deepsense import neptune

FLAGS = tf.app.flags.FLAGS


def main():
    ctx = neptune.Context()
    ctx.integrate_with_tensorflow()
    FLAGS.max_steps = 2001
    cifar10_train.main()
    submission, true_labels = cifar10_submission.main()
    tags = ["tensorflow", "tutorial"]
    evaluation.evaluate_and_send_to_neptune(submission, true_labels, ctx, tags)

if __name__ == '__main__':
    main()

