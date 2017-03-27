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

library(xgboost)
library(neptune)
library(ModelMetrics)

# Bank Marketing dataset from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
customer_data <- read.csv('https://s3-us-west-2.amazonaws.com/deepsense.neptune/data/bank-additional/bank-additional-full.csv', sep = ';')

# Some preprocessing
customer_data$duration <- NULL
y <- customer_data$y == 'yes'
x <- model.matrix(y~.-1, data = customer_data)

# Train, validation and test sets
set.seed(999)
train_valid_test <- sample(1:3, prob = c(0.6, 0.2, 0.2), replace = T, size = nrow(x))
train_idx <- train_valid_test == 1
valid_idx <- train_valid_test == 2
test_idx <- train_valid_test == 3

y_train <- y[train_idx]
y_valid <- y[valid_idx]
y_test <- y[test_idx]
x_train <- x[train_idx,]
x_valid <- x[valid_idx,]
x_test <- x[test_idx,]

# Setup channels and plots
createNumericChannel('train_auc')
createNumericChannel('valid_auc')
createNumericChannel('test_auc')
createChart(chartName = 'Train & validation auc', series = list('train_auc', 'valid_auc'))
createNumericChannel('execution_time')
createChart(chartName = 'Total execution time', series = list('execution_time'))

cb.print.evaluation <- function (period = 1)  {
  callback <- function(env = parent.frame()) {
    if (length(env$bst_evaluation) == 0 || period == 0)
      return()
    i <- env$iteration
    if ((i - 1)%%period == 0 || i == env$begin_iteration ||
        i == env$end_iteration) {
      channelSend('train_auc', i, env$bst_evaluation[1])
      channelSend('valid_auc', i, env$bst_evaluation[2])
      channelSend('execution_time', i, as.numeric(Sys.time() - start_time))
    }
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.print.evaluation'
  callback
}

# Training Xgboost model
nrounds <- params('nrounds')
start_time <- Sys.time()
model <- xgb.train(
  params = list(
    objective = 'binary:logistic', 
    eval_metric = 'auc',
    max_depth = 4),
  data = xgb.DMatrix(x_train, label = y_train),
  nrounds = nrounds,
  watchlist = list(
    train = xgb.DMatrix(x_train, label = y_train),
    validation = xgb.DMatrix(x_valid, label = y_valid)),
  callbacks = list(cb.print.evaluation()))

# Evaluation using test set
predictions_test <- predict(model, xgb.DMatrix(x_test), ntreelimit = 10)
auc_test <- auc(y_test, predictions_test)
channelSend('test_auc', 1, auc_test)

lift_chart <- function(responses, predictions) {
  baseline <- mean(responses)
  responses_ordered <- responses[order(predictions, decreasing = TRUE)]
  lift <- cumsum(responses_ordered) / 1:length(responses_ordered) / baseline
  createNumericChannel('lift')
  createChart(chartName = 'Lift chart', series = list('lift'))
  n <- length(lift)
  for(x in seq(0.1, 1, by = 0.1)) {
    # max(., 1) assures a proper index >= 1
    channelSend('lift', x, lift[max(round(x * n), 1)])
  }
}

lift_chart(y_test, predictions_test)

