
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf



FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  print("some objects format")

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  print("image (placeholder) shape: " + str( x.get_shape() ) )
  W = tf.Variable(tf.zeros([784, 10]))
  print("W (variabile) shape: " + str( W.get_shape() ) )
  b = tf.Variable(tf.zeros([10]))
  print("bias (variabile) shape: " + str( b.get_shape() ) )
  y = tf.matmul(x, W) + b
  print("classes shape: " + str( y.get_shape() ) )


  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  print("y_, that is the true answer, shape: " + str( y_.get_shape() ) )

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  print("cross entropy defined as a function, so no dimensions: " + str( cross_entropy.get_shape() ) )
  
  train_step = tf.train.GradientDescentOptimizer(0.25).minimize(cross_entropy)

  print("train_step defined as an operation: " )
  
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for k in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if( k % 100 == 0):
      #print("image (input) shape in this batch: " + str( batch_xs[0] ) )
      print("true-results (input) shape in this batch: " + str( batch_ys[0] ) )
      print( sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
