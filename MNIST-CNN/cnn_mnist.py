# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:27:43 2017

Tensorflow - Convolutional Neural Network

@author: Daniel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic
def cnn_model_fn(features, labels, mode):
    
  # Input Layer
  input_layer = tf.reshape(features, [-1, 28, 28, 1]) # [batch_size, image_width, image_height, channels]

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d( # Output: [batch_size, 28, 28, 32]
      inputs=input_layer,
      filters=32, # Number of filters
      kernel_size=[5, 5], # [filter_width, filter_height]
      padding="same", # Adds 0s to the edges to produce feature maps of dimensions 28x28 instead of 24x24 (stride = 1)
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d( # Output: [batch_size, 14, 14, 32]
          inputs=conv1, 
          pool_size=[2, 2], # [window_width, window_height]
          strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d( # Output: [batch_size, 14, 14, 64]
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d( # Output: [batch_size, 7, 7, 64]
          inputs=conv2, 
          pool_size=[2, 2], 
          strides=2)

  # Dense Layer
  
  ## Flatten feature maps
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # Transform 4 dimensional to a tensor of 2 dimensions (Output: [batch_size, 3136])
  
  ## Dense layer
  dense = tf.layers.dense( # Output: [batch_size, 1024]
          inputs=pool2_flat, 
          units=1024, 
          activation=tf.nn.relu)
  
  ## Dropout regularization
  dropout = tf.layers.dropout( # Output: [batch_size, 1024]
          inputs=dense,
          rate=0.4, # 40% of the elements are randomly dropped out during training
          training=mode == learn.ModeKeys.TRAIN) # Only performed in the case of training mode

  # Dense layer (Logits Layer)
  logits = tf.layers.dense( # Output: [batch_size, 10] (predictions)
          inputs=dropout, 
          units=10)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = None
  if mode != learn.ModeKeys.INFER:
      
    ## Convert labels (containing a list of predictions for the examples) to one-hot encoding 
    onehot_labels = tf.one_hot(
            indices=tf.cast(labels, tf.int32), # Locations in the one-hot tensor that will have "on values" (1) 
            depth=10) # Number of target classes
    
    ## Calculate cross entropy of one-hot labels and the softmax of the predictions (logits)
    loss = tf.losses.softmax_cross_entropy( # Output: error
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  train_op = None
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, 
          axis=1),
      "probabilities": tf.nn.softmax(
          logits, 
          name="softmax_tensor")
  }

  # Return a ModelFnOps object
  model = model_fn_lib.ModelFnOps(
          mode=mode, 
          predictions=predictions, 
          loss=loss, 
          train_op=train_op)
  
  return model
ta = mnist.train.images # Returns np.array
  train_l
# Train and evaluate classifier
def main(unused_argv):
    
  # Load training and eval data
  mnist = learn.datasets.load_dataset("mnist")
  train_daabels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  # Create the Estimator
  mnist_classifier = learn.Estimator(      
          model_fn=cnn_model_fn, 
          model_dir="/tmp/mnist_convnet_model")
  
if __name__ == "__main__":
  tf.app.run()

