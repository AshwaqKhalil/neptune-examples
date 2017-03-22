# Neptune Examples - Python

## Getting Started
The goal of this example is to create a simple parameterizable job
that generates sine and cosine as functions of time (in seconds),
with the provided `amplitude` and `sampling_rate`.
The full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/getting-started.html">documentation</a>.

### Run Command

    cd getting-started
    neptune run

## Examples

### Handwritten Digits Recognition
This example is an adaptation of source code from the deep learning
<a target="_blank" href="https://keras.io/">Keras</a>
library which shows utilization of Neptune features.
The example consists of a single Python file and uses Keras
to train and evaluate a convolutional neural network that recognizes handwritten digits.
Full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/examples/handwritten-digits-recognition.html">documentation</a>.

#### Additional Requirements

* <a target="_blank" href="https://keras.io/">Keras</a>

#### Run Command

    cd handwritten-digits-recognition
    neptune run mnist_cnn_neptune.py -- --kernel_size 5

### Diabetes Progression Prediction
This example is an adaptation of
<a target="_blank" href="http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html">Linear Regression example</a>
from the <a target="_blank" href="http://scikit-learn.org/stable/">scikit-learn</a>
machine learning library which shows utilization of [Grid Search](https://neptune.deepsense.io/versions/latest/reference-guides/job-and-experiment.html#experiments).
The example consists of a single Python file using scikit-learn to train and evaluate a simple linear regression model
that predicts disease progression of diabetes patients.
The full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/examples/diabetes-progression-prediction.html">documentation</a>.

#### Additional Requirements

* <a target="_blank" href="http://scikit-learn.org/stable/install.html">scikit-learn 0.17</a>
* <a target="_blank" href="http://matplotlib.org/users/installing.html">Matplotlib 1.5.3</a>

#### Run Command

    cd diabetes-progression-prediction
    neptune run plot_ols_neptune.py -- --feature_index 2

### Flower Species Prediction
This example is an adaptation of
<a target="_blank" href="https://www.tensorflow.org/versions/r0.11/tutorials/tflearn/index.html">Quick Start</a>
from the <a target="_blank" href="https://www.tensorflow.org/">TensorFlow</a>
library, which shows the ease of integration between the two platforms.
The example consists of a single Python file using TensorFlow to train and evaluate a simple
neural network that predicts flower species (Iris).
The full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/examples/flower-species-prediction.html">documentation</a>.

#### Additional Requirements

* <a target="_blank" href="https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup">TensorFlow 0.11.0</a>

#### Run Command

    cd flower-species-prediction
    neptune run

### Leaderboard
This example is an adaptation of
<a target="_blank" href="https://www.tensorflow.org/tutorials/deep_cnn/">Convolutional Neural Networks</a>
from the <a target="_blank" href="https://www.tensorflow.org/">TensorFlow</a>
library, which shows the ease of adding Neptune features to the code.
The example consists of Python files using TensorFlow to train and evaluate a convolutional neural network that predicts image categories (CIFAR-10). Then, the evaluated metrics are sent to Neptune to build the leaderboard.

#### Additional Requirements

* <a target="_blank" href="https://www.tensorflow.org/install/">TensorFlow 1.0.1</a>
* TensorFlow models repository in `PYTHONPATH`

You should setup the TensorFlow models repository with the following commands:

    git clone https://github.com/tensorflow/models/
    export PYTHONPATH="$PWD/models/tutorials/image/cifar10:$PYTHONPATH"

#### Run Command

    cd leaderboard
    neptune run

### RoI pooling
This example presents RoI pooling in <a target="_blank" href="https://www.tensorflow.org/">TensorFlow</a> based on our <a target="_blank" href="https://github.com/deepsense-io/roi-pooling">custom RoI pooling TensorFlow operation</a>.
The network with the <a target="_blank" href="https://arxiv.org/pdf/1504.08083.pdf">Fast R-CNN</a> architecture detects cars on images.

#### Additional Requirements

* <a target="_blank" href="https://developer.nvidia.com/cuda-downloads">CUDA 8</a>
* <a target="_blank" href="https://www.tensorflow.org/">TensorFlow 1.0</a> with GPU support
* our <a target="_blank" href="https://github.com/deepsense-io/roi-pooling">custom RoI pooling TensorFlow operation</a>
* <a target="_blank" href="http://opencv.org/">OpenCV</a>

You should setup the custom RoI pooling TensorFlow operation with the following commands:

    git clone https://github.com/deepsense-io/roi-pooling
    python setup.py install

You also should download the file `vgg16-20160129.tfmodel` referred to by the torrent file `vgg16-20160129.tfmodel.torrent`
and save it in the `data` directory.

#### Run command
    cd roi-pooling
    export DATA_DIR=`pwd`/data
    cd code
    neptune run \
    -- \
    --im_folder $DATA_DIR/images \
    --roidb $DATA_DIR/roidb \
    --pretrained_path $DATA_DIR/vgg16-20160129.tfmodel

