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

def testDNNRegressor( training_set, test_set, prediction_set, logs_path ):
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]

	DNNRegressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
	                                   hidden_units=[10, 10, 10],
	                                   model_dir = logs_path,
	                                   optimizer=tf.train.ProximalAdagradOptimizer(
									      learning_rate=0.01,
									      l1_regularization_strength=0.0005
									    ))



	# Train
	DNNRegressor.fit(x=training_set.data,
	               y=training_set.target,
	               steps=10000)

	# Evaluate accuracy.
	accuracy_score = DNNRegressor.evaluate(x=test_set.data,
	                                     y=test_set.target)['loss']


	tf.summary.scalar("loss", accuracy_score)

	print("\n\nTOTAL LOSS: " + str( accuracy_score ) )


	y = list(DNNRegressor.predict(prediction_set.data, as_iterable=True))


	print( "\n#################################### DNN REGRESSOR RESULTS ##############################" )
	for i in range( 0 , len(y) ):
		print( "Predicted:  " + str( y[i] ) + ";     Real: " +  str( prediction_set.target[i] )  )

	print ("\n")	

def testLinearRegressor( training_set, test_set, prediction_set, logs_path ):

	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]

	LinearRegressor = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns)

	# Train
	LinearRegressor.fit(x=training_set.data,
	               y=training_set.target,
	               steps=3500)

	# Evaluate accuracy.
	accuracy_score = LinearRegressor.evaluate(x=test_set.data,
	                                     y=test_set.target)['loss']

	print("\n\nTOTAL LOSS: " + str( accuracy_score ) )


	y = list(LinearRegressor.predict(prediction_set.data, as_iterable=True))


	print( "\n#################################### LINEAR REGRESSOR RESULTS ##############################" )
	for i in range( 0 , len(y) ):
		print( "Predicted:  " + str( y[i] ) + ";     Real: " +  str( prediction_set.target[i] )  )

	print ("\n")	


def testDNNLinearCombinedRegressor( training_set, test_set, prediction_set ):

	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]

	combined = tf.contrib.learn.DNNLinearCombinedRegressor(
	    # wide settings
	    linear_feature_columns=feature_columns,
	    dnn_feature_columns=feature_columns,
	    dnn_hidden_units=[30, 20, 10])
	

	# Train
	combined.fit(x=training_set.data,
	               y=training_set.target,
	               steps=1500)

	# Evaluate accuracy.
	accuracy_score = combined.evaluate(x=test_set.data,
	                                     y=test_set.target)['loss']

	print("\n\nTOTAL LOSS: " + str( accuracy_score ) )


	y = list(combined.predict(prediction_set.data, as_iterable=True))


	print( "\n############################# COMBINED REGRESSOR-CLASSIFIED RESULTS ########################" )
	for i in range( 0 , len(y) ):
		print( "Predicted:  " + str( y[i] ) + ";     Real: " +  str( prediction_set.target[i] )  )

	print ("\n")	




FLAGS = None
logs_path = "/home/andrea/Desktop/python/tensorflowTutorialGenerical/logs/"
tf.logging.set_verbosity(tf.logging.INFO)

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


#testDNNEstimator( training_set, test_set, prediction_set )
testDNNRegressor( training_set, test_set, prediction_set, logs_path )
#testLinearRegressor( training_set, test_set, prediction_set )
#testDNNLinearCombinedRegressor( training_set, test_set, prediction_set )


print("\n Finished")