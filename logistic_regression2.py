# a logistic regression learning algorithm example using TensorFlow library
# this example is using the MNIST database of handwritten digits
from __future__ import print_function
import sys
sys.path.append('/opt/tensorflow/work/tutorials/mnist')

import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf graph input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28 = 784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x,W) + b) # softmax

# minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})
      
            # compute average cost
            avg_cost += c / total_batch

        # display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # test model 
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
