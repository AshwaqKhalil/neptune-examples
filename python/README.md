# Neptune Examples - Python

## Getting Started
The goal of this example is to create a simple parameterizable job
that generates sine and cosine as functions of time (in seconds),
with the provided `amplitude` and `sampling_rate`.
The full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/getting-started.html">documentation</a>.

### Run Command

    cd getting-started
    neptune run main.py --config config.yaml --dump-dir-url my_dump_dir

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

    neptune run mnist_cnn_neptune.py --config config.yaml --dump-dir-url mnist_cnn_neptune_output -- --kernel_size 5

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

    neptune run plot_ols_neptune.py --config config.yaml --dump-dir-url plot_ols_neptune_output -- --feature_index 2

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

    neptune run flower-species-prediction/main.py --config flower-species-prediction/config.yaml --storage-url /tmp/neptune-iris --paths-to-dump flower-species-prediction

### Leaderboard
This example is an adaptation of
<a target="_blank" href="https://www.tensorflow.org/tutorials/deep_cnn/">Convolutional Neural Networks</a>
from the <a target="_blank" href="https://www.tensorflow.org/">TensorFlow</a>
library, which shows the ease of adding Neptune features to the code.
The example consists of Python files using TensorFlow to train and evaluate a convolutional neural network that predicts image categories (CIFAR-10). Then, the evaluated metrics are sent to Neptune to build the leaderboard.

#### Additional Requirements

* <a target="_blank" href="https://www.tensorflow.org/install/">TensorFlow 1.0.1</a>
* Tensorflow models repository in `PYTHONPATH`

You should setup the Tensorflow models repository with the following commands:

    git clone https://github.com/tensorflow/models/
    export PYTHONPATH="$PWD/models/tutorials/image/cifar10:$PYTHONPATH"

#### Run Command

    neptune run leaderboard/main.py --config leaderboard/config.yaml --dump-dir-url leaderboard/dump --paths-to-dump leaderboard
