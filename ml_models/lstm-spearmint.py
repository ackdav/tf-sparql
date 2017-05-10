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
state_size = 52#66
num_timesteps = 1#4 #How many steps to look back
num_classes = 1


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

def recurrent_neural_network(num_layers, init_state, name="rnn-net"):
	'''Building the network and returning it's final output and current state

	Returns:
		output: Output of the network created
		current_state: Current hidden state of the network of size [num_layers, 2, batch_size, state_size]
	'''
	with tf.name_scope(name) as scope:
		
		layer = {'weights':tf.get_variable(shape=[state_size,num_classes], initializer=tf.contrib.layers.xavier_initializer(), name="w"),
					'biases':tf.get_variable(shape=[num_classes], initializer=tf.contrib.layers.xavier_initializer(), name="b")}

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

def train_rnn(learning_rate, benchmark_err, batch_size, num_layers, early_stop_flag, log_param="cold_old_3"):
	'''Trains and tests the RNN network.

	Args:
		benchmark_err: 'tuple', the relative mean error of the database for Y_train and Y_test
		log_param: 'string', used for saving logs for the current run
	'''
	begin_time = time.time()
	init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
	prediction, current_state = recurrent_neural_network(num_layers, init_state)

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
		last_n_results = []
		results = []
		final_res = 0.

		e_opt = 0.
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

				if epoch == 0:
					e_opt = avg_cost
				early_stop = 100 * ((avg_cost/e_opt)-1)
				if early_stop > early_stop_flag and epoch > 10:
					break
				if avg_cost < e_opt:
					e_opt = avg_cost


				if len(last_n_results) > 4:
					del last_n_results[-1]
				last_n_results.insert(0, avg_cost)
				results.append(avg_cost)
				end_res = sum(last_n_results) / len(last_n_results)
				final_res = end_res
				# print ("Error with model: {0:.6f} ### ".format(avg_cost)),
				# print ("Error no model: {0:.6f}".format(benchmark_err[1]))
				# results.append(avg_cost)
			#==========test model============

			#==========training model============
			epoch_start = time.time()
			for i in range(total_batch-1):
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]

				batch_x = batch_x.reshape([batch_size, num_timesteps, state_size])

				batch_y = np.transpose([batch_y])
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c, p, s_, _current_state = sess.run([optimizer_train, perc_err_train, prediction, summary_op, current_state], 
											feed_dict={x: batch_x, y: batch_y, init_state: _current_state})
				
				loss_vec.append(c)
				# Compute average loss
				# avg_cost += c / total_batch
				writer.add_summary(s_, epoch)
				# avg_cost_vec.append(avg_cost)
			#==========training model============
		return final_res

			#Throughput queries / second
			# epoch_end = time.time()-epoch_start
			# throughput = len(X_train) / epoch_end
			# throughputs.append(throughput)
			# throughput_avg = sum(throughputs) / len(throughputs)			

			#==========training log============
	 	# 	label_value = batch_y
	 	# 	estimate = p
	 	# 	err = label_value-estimate
			# # Display logs per epoch step
			# if epoch % 1 == 0:
			# 	print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c), "throughput=","{:.4f}".format(throughput_avg))
			# 	print ("[*]----------------------------")
			# 	for i in xrange(4):
			# 		print ("label value:", label_value[i], \
			# 			"estimated value:", estimate[i])
			# 	print ("[*]============================")
			# 	sys.stdout.flush()
			#==========training log============

			# if epoch % 99 == 0 and epoch != 0:
			#     with open('results-rnn.txt', 'a') as out_:
			#         out_.write('structure' + str(results) + '\n')

		# print("Total Time: %3.2fs" % float(time.time() - begin_time))
		# print("Throughput Avg: %.4f" % float(throughput_avg))

def main(job_id, params):
	warm = False
	vector_options = {'structure': True,'time': False, 'ged': False,'sim': False,'w2v': False}

	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input, state_size
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('database-iMac.log', warm, vector_options)

	benchmark_err = no_modell_mean_error(Y_train, Y_test)
	# print benchmark_err
	# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parser.add_argument('--model', type=str, default='gru',
	#                 help='rnn, gru, lstm, or nas')
	# parser.add_argument('--num_layers', type=int, default=2,
	#                 help='number of layers in the RNN')
	# args = parser.parse_args()

	# train_rnn(benchmark_err)
	return train_rnn(params['learning_rate'][0], benchmark_err,  params['batch_size'][0], params['num_layers'][0],params['gl'][0])


if __name__ == '__main__':
	main()