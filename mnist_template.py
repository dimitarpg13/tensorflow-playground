# a logistic regression learning algorithm example using TensorFlow library
# this example is using the MNIST database of handwritten digits
from __future__ import print_function
import sys
sys.path.append('/opt/tensorflow/work/tutorials/mnist')

import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


