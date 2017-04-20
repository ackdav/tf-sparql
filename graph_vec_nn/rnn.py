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
from plstm.PhasedLSTMCell_v1 import PhasedLSTMCell, multiPLSTM

LOGDIR = 'logs/rnn/'

hm_epochs = 100
batch_size = 16
state_size = 24#66
num_timesteps = 1#4 #How many steps to look back
num_classes = 1
num_layers = 3

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
test_loss = []
avg_cost_vec = []

def recurrent_neural_network(x, args, name="rnn-net"):
	'''Executes the RNN model

	Keyword arguments:
	x -- input x, initially a placeholder, but in a session will contain features
	args -- cmd-line arguments
	'''
	with tf.name_scope(name):
		layer = {'weights':tf.Variable(tf.random_normal([state_size,num_classes])),
						'biases':tf.Variable(tf.random_normal([num_classes]))}
						
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
		#outputs gives all states, but we only want last one
		output = tf.matmul(last, layer['weights']) + layer['biases']
		return (output, current_state)

def train_neural_network(x, benchmark_err, args, log_param="cold_similarity"):
	global X_test
	# _current_state = [None for _ in range(num_layers)]
	prediction, current_state = recurrent_neural_network(x, args)

	begin_time = time.time()
	# cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	with tf.name_scope("relativeMeanError"):
		perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err))
		tf.summary.scalar("relativeMeanError", perc_err)
	with tf.name_scope("optimizer"):
		optimizer = tf.train.AdamOptimizer(0.0001).minimize(perc_err)

	summary_op = tf.summary.merge_all()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

		for epoch in range(hm_epochs):
			temp_loss = 0.
			# avg_cost = 0.
			_current_state = np.zeros((num_layers, 2, batch_size, state_size))
			total_batch = num_training_samples//batch_size

			for i in range(total_batch-1):
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]
				
				batch_x = batch_x.reshape([batch_size, num_timesteps, state_size])

				batch_y = np.transpose([batch_y])
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c, p, s_, _current_state = sess.run([optimizer, perc_err, prediction, summary_op, current_state], 
											feed_dict={x: batch_x, y: batch_y, init_state: _current_state})
				loss_vec.append(c)
				# Compute average loss
				# avg_cost += c / total_batch
				writer.add_summary(s_, epoch)
				# avg_cost_vec.append(avg_cost)

			# sample prediction
	 		label_value = batch_y
	 		estimate = p
	 		err = label_value-estimate

			# Display logs per epoch step
			if epoch % 1 == 0:

				print ("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(c))
				print ("[*]----------------------------")
				for i in xrange(4):
					print ("label value:", label_value[i], \
						"estimated value:", estimate[i])
				print ("[*]============================")
				sys.stdout.flush()
			if epoch % 10 == 0:
				num_test_batches = X_test.shape[0]//batch_size
				test_state = np.zeros((num_layers, 2, batch_size, state_size))
				# _current_state = [None for _ in range(num_layers)]
				avg_cost = 0.
				for i in range(num_test_batches):
					batch_x_test = X_test[i*batch_size:(i+1)*batch_size]
					batch_y_test = Y_test[i*batch_size:(i+1)*batch_size]
					batch_x_test = batch_x_test.reshape([batch_size, num_timesteps, state_size])
					# X_test = X_test.reshape([X_test.shape[0], num_timesteps, state_size])
					_, c, s_, test_state = sess.run([optimizer, perc_err, summary_op, current_state], 
											feed_dict={x: batch_x_test, y: batch_y_test, init_state: test_state})
					avg_cost += c / num_test_batches
				# print ("relative error with model: {0:.3f}".format(perc_err.eval({x: X_test, y: Y_test, init_state: test_state})))
				print ("relative error with model: {0:.3f}".format(avg_cost))
				print ("relative error without model: {0:.3f}".format(benchmark_err))
		print("Total Time: %3.2fs" % float(time.time() - begin_time))

def main():
	warm = False
	vector_options = {'structure': False,'time': False, 'ged': False,'sim': True,'w2v': False}
	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input, state_size
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('database.log-complete', warm, vector_options)
	# X_test, Y_test = adjust_rnn_test_arrays(X_test, Y_test, num_timesteps, state_size)
	# state_size = len(X_train[0])
	benchmark_err = no_modell_mean_error(Y_train, Y_test)
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default='gru',
                    help='rnn, gru, lstm, or nas')
	parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
	args = parser.parse_args()

	train_neural_network(x,benchmark_err, args)

if __name__ == '__main__':
	main()