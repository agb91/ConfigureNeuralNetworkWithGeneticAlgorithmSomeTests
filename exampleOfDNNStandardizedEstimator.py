
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.contrib.learn.python.learn.estimators.dnn import DNNEstimator
from tensorflow.contrib.learn.python.learn.estimators.dnn import DNNRegressor
import tempfile

from six.moves import urllib
import pandas as pd
import numpy as np
import tensorflow as tf
from gene import Gene

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["a","b","c","d","e","f","g","target"]
FEATURES = ["a","b","c","d","e","f","g"]
LABEL = "target"

path = "/home/andrea/Desktop/python/tensorflowTutorialGenerical/"
abalone_train = path + "headedTraining"
abalone_test = path + "test"
abalone_predict = path + "toPredictWithElder"

# Training examples
training_set = pd.read_csv(abalone_train, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
training_set_features = training_set.ix[:,0:7]
training_set_labels = training_set.ix[:,7]

#print( training_set_labels )


# Test examples
test_set = pd.read_csv(abalone_test, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set_features = test_set.ix[:,0:7]
test_set_labels = test_set.ix[:,7]

# Set of 7 examples for which to predict abalone ages
prediction_set = pd.read_csv(abalone_predict, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
prediction_set_features = prediction_set.ix[:,0:7]
prediction_set_labels = prediction_set.ix[:,7]


feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

nn = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                hidden_units=[10,10],
                                model_dir=path + "/models")



# Train
nnInput = tf.estimator.inputs.pandas_input_fn(
      x= pd.DataFrame( training_set_features),
      y = pd.Series(training_set_labels),
      num_epochs=None,
      shuffle=True)

nn.train(input_fn=nnInput, steps=5000)



# Evaluate
nnTesting = tf.estimator.inputs.pandas_input_fn(
      x= pd.DataFrame( test_set_features),
      y = pd.Series(test_set_labels),
      num_epochs=1,
      shuffle=False)

ev = nn.evaluate(
    input_fn = nnTesting )

print( ev['average_loss'] )








