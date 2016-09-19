# Neptune Examples

## Overview

Examples for
<a target="_blank" href="https://neptune.deepsense.io">Neptune</a>.

To run the following examples you need to have
<a target="_blank" href="https://neptune.deepsense.io/latest/downloads.html">Neptune CLI</a>
installed.

## Python Examples

### Getting Started
The goal of this example is to create a simple parameterizable job
that generates sine and cosine as functions of time (in seconds),
with the provided `amplitude` and `sampling_rate`.
The full description of the example can be found in the
<a target="_blank" href="https://neptune/deepsense.io/versions/latest/getting-started.html">documentation</a>.

#### Run Command

    neptune run main.py --config config.yaml --dump-dir-url my_dump_dir

### Handwritten Digits Recognition
This example is an adaptation of source code from the deep learning
<a target="_blank" href="https://keras.io/">Keras</a>
library which shows utilization of Neptune features.
The example consists of a single Python file and uses Keras
to train and evaluate a convolutional neural network that recognizes handwritten digits.
Full description of the example can be found in the
<a target="_blank" href="https://neptune/deepsense.io/versions/latest/examples/handwritten-digits-recognition.htm">documentation</a>.

#### Additional Requirements

* <a target="_blank" href="https://keras.io/">Keras</a>

#### Run Command

    neptune run mnist_cnn_neptune.py --config config.yaml --dump-dir-url mnist_cnn_neptune_output -- --kernel_size 5

