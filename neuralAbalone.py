#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""DNNRegressor with custom estimator for abalone dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf

class neuralAbalone():
  '''
  FLAGS = None

  tf.logging.set_verbosity(3)

  # Learning rate for the model
  LEARNING_RATE = 0.01
  # How many steps
  STEPS = 5000
  #How many units in each layer? (only if the layer exists obvious..)
  UNITS1 = 10
  UNITS2 = 10
  UNITS3 = 10
  # if 1 there is a second layer
  SECOND = 1
  # if 1 there is a third layer
  THIRD = 0

  SET_OF_FEATURES = [0,1,2,3,5,6]
  '''
  def __init__(self , LEARNING_RATE, STEPS, UNITS1, UNITS2, UNITS3, SECOND, THIRD, SET_OF_FEATURES):
    self.LEARNING_RATE = LEARNING_RATE 
    self.STEPS = STEPS
    self.UNITS1 = UNITS1
    self.UNITS2 = UNITS2
    self.UNITS3 = UNITS3
    self.SECOND = SECOND
    self.THIRD = THIRD  
    self.SET_OF_FEATURES = SET_OF_FEATURES

  def model_fn(self, features, labels, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
    first_hidden_layer = tf.layers.dense(features["x"], self.UNITS1, activation=tf.nn.relu)

    if(self.SECOND == 1):
      # Connect the second hidden layer to first hidden layer with relu
      second_hidden_layer = tf.layers.dense(
        first_hidden_layer, self.UNITS2, activation=tf.nn.relu)
    else:
      second_hidden_layer = first_hidden_layer

    if(self.THIRD == 1):
      # Connect the third hidden layer to second hidden layer with relu
      third_hidden_layer = tf.layers.dense(
        second_hidden_layer, self.UNITS3, activation=tf.nn.relu)
    else:
      third_hidden_layer = second_hidden_layer
    
    # Connect the output layer to third hidden layer (no activation fn)
    output_layer = tf.layers.dense(third_hidden_layer, 1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"ages": predictions})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float64), predictions)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


  def analyzeAbalones( self ):
    # Load datasets
    path = "/home/andrea/Desktop/python/tensorflowTutorialGenerical/"
    abalone_train = path + "training"
    abalone_test = path + "test"
    abalone_predict = path + "toPredictWithElder"

    # Training examples
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

    training_set_to_use = training_set.data[:, self.SET_OF_FEATURES ]
    
    # Test examples
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

    test_set_to_use = test_set.data[:, self.SET_OF_FEATURES]

    # Set of 7 examples for which to predict abalone ages
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

    prediction_set_to_use = prediction_set.data[:, self.SET_OF_FEATURES]
    # Set model params
    model_params1 = {"learning_rate": self.LEARNING_RATE}


    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=self.model_fn, params=model_params1)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set_to_use)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train
    nn.train(input_fn=train_input_fn, steps=self.STEPS)

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set_to_use)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    #print("Loss: %s" % ev["loss"])
    #print("Root Mean Squared Error: %s" % ev["rmse"])

    # Print out predictions
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": prediction_set_to_use},
        num_epochs=1,
        shuffle=False)
    predictions = nn.predict(input_fn=predict_input_fn)
    for i, p in enumerate(predictions):
      print("Prediction %s: %s  -----  real was:  %s" % (i + 1, p["ages"] , prediction_set[1][i]) )



  def run(self):
    self.analyzeAbalones()

