# Neptune Examples - R

## Getting Started
The goal of this example is to create a simple parameterizable job
that generates sine and cosine as functions of time (in seconds),
with the provided `amplitude` and `sampling_rate`.
The full description of the example can be found in the
<a target="_blank" href="https://neptune.deepsense.io/versions/latest/getting-started.html">documentation</a>.

### Run Command

    cd getting-started
    neptune run

## Bank Marketing Dataset
In this example a dataset of
<a target="_blank" href="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing">bank customers</a> is analyzed. The target variable that we want to predict is whether a customer subscribes to a bank deposit. In order to build the model we use <a target="_blank" href="https://xgboost.readthedocs.io/en/latest/Xgboost">XGBoost</a> - a powerful library for building ensemble models using the Gradient Boosting algorithm. Neptune is used here for creating charts during model training and evaluation and identification of an adequate early stopping moment.

### Dataset Credits

Moro, S., Cortez, P. and Rita, P. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62:22-31, 2014.

Lichman, M. UCI Machine Learning Repository http://archive.ics.uci.edu/ml. Irvine, CA: University of California, School of Information and Computer Science, 2013.

### Additional Requirements

R libraries:

* <a target="_blank" href="https://cran.r-project.org/web/packages/xgboost/">Xgboost</a>
* <a target="_blank" href="https://cran.r-project.org/web/packages/ModelMetrics/">ModelMetrics</a>

### Run Command

    cd bank-marketing
    neptune run bank_marketing.R --nrounds 50 --config config.yaml --dump-dir-url my_dump_dir
