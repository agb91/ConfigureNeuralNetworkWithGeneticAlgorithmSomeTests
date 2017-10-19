from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# Import urllib
from six.moves import urllib

import numpy as np
import tensorflow as tf


def testDNNEstimator( training_set, test_set, prediction_set ):
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]

	DNNEstimator = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
	                                   hidden_units=[7, 15, 32],
	                                   activation_fn=tf.nn.relu,
	                                   dropout=0.2,
	                                   n_classes=30,
	                                   optimizer="Adam")


	# Train
	DNNEstimator.fit(x=training_set.data,
	               y=training_set.target,
	               steps=1500)

	# Evaluate accuracy.
	accuracy_score = DNNEstimator.evaluate(x=test_set.data,
	                                     y=test_set.target)['loss']
	print("\n\nTOTAL LOSS: " + str( accuracy_score ) )


	y = list(DNNEstimator.predict(prediction_set.data, as_iterable=True))


	print( "\n#################################### DNN ESTIMATOR RESULTS ##############################" )
	for i in range( 0 , len(y) ):
		print( "Predicted:  " + str( y[i] ) + ";     Real: " +  str( prediction_set.target[i] )  )

	print ("\n")	

def testDNNRegressor( training_set, test_set, prediction_set ):
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]

	DNNRegressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
	                                   hidden_units=[10, 5],
	                                   optimizer=tf.train.ProximalAdagradOptimizer(
									      learning_rate=0.1,
									      l1_regularization_strength=0.001
									    ))


	# Train
	DNNRegressor.fit(x=training_set.data,
	               y=training_set.target,
	               steps=1500)

	# Evaluate accuracy.
	accuracy_score = DNNRegressor.evaluate(x=test_set.data,
	                                     y=test_set.target)['loss']

	print("\n\nTOTAL LOSS: " + str( accuracy_score ) )


	y = list(DNNRegressor.predict(prediction_set.data, as_iterable=True))


	print( "\n#################################### DNN REGRESSOR RESULTS ##############################" )
	for i in range( 0 , len(y) ):
		print( "Predicted:  " + str( y[i] ) + ";     Real: " +  str( prediction_set.target[i] )  )

	print ("\n")	


FLAGS = None

path = "/home/andrea/Desktop/python/tensorflowTutorialGenerical/"
abalone_train = path + "training"
abalone_test = path + "test"
abalone_predict = path + "toPredictWithElder"

# Training examples
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

# Test examples
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

# Set of 7 examples for which to predict abalone ages
prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)


testDNNEstimator( training_set, test_set, prediction_set )
#testDNNRegressor( training_set, test_set, prediction_set )


print("\n Finished")