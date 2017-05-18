from __future__ import division

import tensorflow as tf
tf.reset_default_graph()

import re
import ast
import time
import argparse
import itertools
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from nn_helper import *

LOGDIR = 'logs/rnn/'

hm_epochs = 100
batch_size = 16
state_size = 62#66
num_timesteps = 1
num_classes = 1
num_layers = 3
learning_rate = 0.001832

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
x = tf.placeholder(shape=[None, num_timesteps, state_size], dtype=tf.float32, name="x")
y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name="y")

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
y_vals = np.array([])

# Training loop
loss_vec = []
avg_cost_vec = []

def recurrent_neural_network(name="rnn-net"):
	'''Building the network and returning it's final output and current state

	Returns:
		output: Output of the network created
		current_state: Current hidden state of the network of size [num_layers, 2, batch_size, state_size]
	'''
	with tf.name_scope(name) as scope:
		
		layer = {'weights':tf.get_variable(shape=[state_size,num_classes], initializer=tf.contrib.layers.xavier_initializer(), name="w"),
					'biases':tf.Variable(tf.random_normal(shape=[num_classes], mean=0, stddev=0))}

		#This unpacks the current state placeholer and assigns it
		l = tf.unstack(init_state, axis=0)
		rnn_tuple_state = tuple( [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)])
		
		def lstm_cell():
			cell = tf.contrib.rnn.NASCell(
		      state_size, reuse=tf.get_variable_scope().reuse)
			return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)

		rnn_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple = True)
		outputs, current_state = tf.nn.dynamic_rnn(rnn_cells, x, initial_state=rnn_tuple_state, scope = "layer")

		outputs = tf.transpose(outputs, [1, 0, 2])
		last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
		output = tf.matmul(last, layer['weights']) + layer['biases']
		return (output, current_state)

def train_rnn(benchmark_err, log_param="cold_old_3"):
	'''Trains and tests the RNN network.

	Args:
		benchmark_err: 'tuple', the relative mean error of the database for Y_train and Y_test
		log_param: 'string', used for saving logs for the current run
	'''
	begin_time = time.time()
	prediction, current_state = recurrent_neural_network()

	with tf.name_scope("relativeMeanError"):
		perc_err_train = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err[0]))
		perc_err_test = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err[1]))
		tf.summary.scalar("relativeMeanError train", perc_err_train)
		tf.summary.scalar("relativeMeanError test", perc_err_test)
	with tf.name_scope("optimizer"):
		optimizer_train = tf.train.AdamOptimizer(learning_rate).minimize(perc_err_train)
		optimizer_test = tf.train.AdamOptimizer(learning_rate).minimize(perc_err_test)

	summary_op = tf.summary.merge_all()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

		results = []
		throughputs = []
		throughput_avg = 0.

		for epoch in range(hm_epochs):
			temp_loss = 0.
			# avg_cost = 0.
			_current_state = np.zeros((num_layers, 2, batch_size, state_size))
			total_batch = num_training_samples//batch_size

			#==========test model============
			if epoch % 2 == 0:
				num_test_batches = X_test.shape[0]//batch_size
				test_state = np.zeros((num_layers, 2, batch_size, state_size))
				# _current_state = [None for _ in range(num_layers)]
				avg_cost = 0.
				for i in range(num_test_batches):
					batch_x_test = X_test[i*batch_size:(i+1)*batch_size]
					batch_y_test = Y_test[i*batch_size:(i+1)*batch_size]
					batch_x_test = batch_x_test.reshape([batch_size, num_timesteps, state_size])
					batch_y_test = np.transpose([batch_y_test])
					# X_test = X_test.reshape([X_test.shape[0], num_timesteps, state_size])
					_, c, s_, test_state = sess.run([optimizer_test, perc_err_test, summary_op, current_state], 
											feed_dict={x: batch_x_test, y: batch_y_test, init_state: test_state})
					avg_cost += c / num_test_batches
				print ("Error with model: {0:.6f} ### ".format(avg_cost)),
				print ("Error no model: {0:.6f}".format(benchmark_err[1]))
				results.append(avg_cost)
			#==========test model============

			#==========training model============
			epoch_start = time.time()
			for i in range(total_batch-1):
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]

				batch_x = batch_x.reshape([batch_size, num_timesteps, state_size])

				batch_y = np.transpose([batch_y])
				# Run optimization op (backprop) and cost op (to get loss value)
				c, p, s_, _current_state = sess.run([perc_err_train, prediction, summary_op, current_state], 
											feed_dict={x: batch_x, y: batch_y, init_state: _current_state})
				
				loss_vec.append(c)
				# Compute average loss
				# avg_cost += c / total_batch
				writer.add_summary(s_, epoch)
				# avg_cost_vec.append(avg_cost)
			#==========training model============

			#Throughput queries / second
			epoch_end = time.time()-epoch_start
			throughput = len(X_train) / epoch_end
			throughputs.append(throughput)
			throughput_avg = sum(throughputs) / len(throughputs)			

			#==========training log============
	 		label_value = batch_y
	 		estimate = p
	 		err = label_value-estimate
			# Display logs per epoch step
			if epoch % 1 == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c), "throughput=","{:.4f}".format(throughput_avg))
				print ("[*]----------------------------")
				for i in xrange(4):
					print ("label value:", label_value[i], \
						"estimated value:", estimate[i])
				print ("[*]============================")
				sys.stdout.flush()
			#==========training log============

		print("Total Time: %3.2fs" % float(time.time() - begin_time))
		print("Throughput Avg: %.4f" % float(throughput_avg))

def main():
	warm = False
	vector_options = {'structure': True,'time': False, 'ged': True,'sim': False,'w2v': False}

	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input, state_size
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('database-iMac.log-complete', warm, vector_options)

	benchmark_err = no_modell_mean_error(Y_train, Y_test)

	train_rnn(benchmark_err)

if __name__ == '__main__':
	main()