'''
largely adapts: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py

goal is to show linear dependency between number of variables in SELECT statement and execution time
'''

from __future__ import print_function

from bio_select_variables import *

import tensorflow as tf
import numpy
import sys, re
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data

train_X = numpy.array([])
train_Y = numpy.array([])

# # Testing example, as requested (Issue #2)
# test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
# test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

n_samples = 0

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

def load_data():
	db = readout_feature()

	global train_X
	global train_Y

	# db_split = re.findall('^(.*?)\n', db, re.DOTALL)
	for entry in (line for i, line in enumerate(db) if i<=250):
		# print(entry)
		entry = re.split(r'[\t|\n]', entry)
		train_X = numpy.append(train_X,float(entry[0]))
		train_Y = numpy.append(train_Y,float(entry[1]))
	return db

def linear_model():
	# Construct a linear model
	return tf.add(tf.mul(X, W), b)

def train_linear_model(data):
	pred = linear_model()

	# Mean squared error
	cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*len(data))

	# Gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Launch the graph
	with tf.Session() as sess:
		# Initializing the variables
		init = tf.global_variables_initializer()

		sess.run(init)

		# Fit all training data
		for epoch in range(training_epochs):
			for (x, y) in zip(train_X, train_Y):
				sess.run(optimizer, feed_dict={X: x, Y: y})

			# Display logs per epoch step
			if (epoch+1) % display_step == 0:
				c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
				# print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
				#     "W=", sess.run(W), "b=", sess.run(b))

		print("Optimization Finished!")
		training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
		# print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

		# Graphic display
		plt.plot(train_X, train_Y, 'ro', label='Original data')
		plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
		plt.legend()
		plt.show()

		# print("Testing... (Mean square loss Comparison)")
		# testing_cost = sess.run(
		# 	tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
		# 	feed_dict={X: test_X, Y: test_Y})  # same function as cost above
		# print("Testing cost=", testing_cost)
		# print("Absolute mean square loss difference:", abs(
		# 	training_cost - testing_cost))

		# plt.plot(test_X, test_Y, 'bo', label='Testing data')
		# plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
		# plt.legend()
		# plt.show()


def main():
	data = load_data()
	train_linear_model(data)


if __name__ == '__main__':
	main()