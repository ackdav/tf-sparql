import tensorflow as tf
tf.reset_default_graph()

import re, ast, time, argparse, itertools
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from nn_helper import *

LOGDIR = 'logs/rnn/'

hm_epochs = 80
batch_size = 10
n_input = 1024#66
state_size = 32#66
num_timesteps = 32#4 #How many steps to look back
num_classes = 1
num_layers = 32

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
	with tf.name_scope(name):
		layer = {'weights':tf.Variable(tf.random_normal([state_size,num_classes])),
						'biases':tf.Variable(tf.random_normal([num_classes]))}
		# x = tf.transpose(x, [1,0,2])
		# x = tf.reshape(x, [-1, state_size])
		# x = tf.split(x, num_timesteps, 0)
		if args.model == 'rnn':
			cell_fn = rnn.BasicRNNCell
		elif args.model == 'gru':
			cell_fn = rnn.GRUCell
		elif args.model == 'lstm':
			cell_fn = rnn.BasicLSTMCell
		elif args.model == 'nas': #only available from TF version 1.1
			cell_fn = rnn.NASCell
		else:
			raise Exception("model type not supported: {}".format(args.model))

		l = tf.unstack(init_state, axis=0)
		rnn_tuple_state = tuple( [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)])
		
		cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
		cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)

		rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
		print(x.shape)
		outputs, current_state = tf.nn.dynamic_rnn(rnn_cells, x, initial_state=rnn_tuple_state)
		# print (outputs.shape, states)
		outputs = tf.transpose(outputs, [1, 0, 2])
		last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
		#outputs gives all states, but we only want last one
		output = tf.matmul(last, layer['weights']) + layer['biases']
		return (output, current_state)


def train_neural_network(x, args, log_param="rnn_no_log_10241"):
	global X_test
	prediction, current_state = recurrent_neural_network(x, args)
	begin_time = time.time()
	# cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	with tf.name_scope("relativeMeanError"):
		perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), tf.reduce_mean(y)))
		tf.summary.scalar("relativeMeanError", perc_err)
	with tf.name_scope("optimizer"):
		optimizer = tf.train.RMSPropOptimizer(0.005).minimize(perc_err)

	# global X_train
	# X_train = X_train.reshape((batch_size, -1))
	summary_op = tf.summary.merge_all()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

		# X_train = X_train.reshape((batch_size, -1))
		# Y_train = Y_train.reshape((batch_size, -1))
		for epoch in range(hm_epochs):
			temp_loss = 0.
			# avg_cost = 0.
			_current_state = np.zeros((num_layers, 2, batch_size, state_size))

			total_batch = num_training_samples//batch_size

			for i in range(total_batch-1):
				
				start_idx = i * num_timesteps
				end_idx = start_idx + batch_size
				# batch_x = x[:,start_idx:end_idx]#RESHAPE
				# batch_y = y[:,start_idx:end_idx]#RESHAPE (other example (batchsize, num_timesteps))
				# batch_x = X_train[start_idx:end_idx]
				# batch_y = Y_train[start_idx:end_idx]
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]

				#reshape to (batch_size, num_timesteps / element )
				# enlarged_batch = []
				
				# for id, obj in enumerate(batch_x):
				# 	try:
				# 		obj = np.append(obj, X_train[i*batch_size+id-1]) 
				# 		obj = np.append(obj, X_train[i*batch_size+id-2])
				# 		obj = np.append(obj, X_train[i*batch_size+id-3])
				# 		# obj = np.append(obj, batch_x[id
				# 		enlarged_batch.append(obj)
				# 	except:
				# 		print "wrong length" + str(id)
				
				# batch_x = np.asarray([enlarged_batch])
				# batch_x=batch_x[0]
				# print(batch_x.shape)
				# batch_x = batch_x.reshape((batch_size, -1))
				# print(batch_x.shape)
				batch_x = batch_x.reshape([batch_size, num_timesteps, state_size])
				# print(batch_x.shape)

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
	 		# print ("num batch:", i)
	 		# perc_err = tf.reduce_mean((batch_y-p)/batch_y)
			# Display logs per epoch step
			if epoch % 1 == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(c))
				print ("[*]----------------------------")
				for i in xrange(5):
					print ("label value:", label_value[i], \
						"estimated value:", estimate[i])
				print ("[*]============================")
				sys.stdout.flush()
			# if epoch % 5 == 0:
			# 	X_test = X_test.reshape([X_test.shape[0], num_timesteps, state_size])
			# 	test_state = np.zeros((num_layers, 8, batch_size, state_size))
			# 	print ("relative error with model: {0:.3f}".format(perc_err.eval({x: X_test, y: Y_test, init_state: test_state})))


		# X_test = X_test.reshape((-1, num_timesteps, state_size))

		# print ("RMSE: {:.3f}".format(cost.eval({x: X_test.reshape((-1, num_timesteps, state_size)), y: Y_test})))
		# print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test.reshape((-1, num_timesteps, state_size)), y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(Y_train, Y_test)))
		print("Total Time: %3.2fs" % float(time.time() - begin_time))


# def plot_result(loss_vec, avg_cost_vec):
# 	# Plot loss (MSE) over time
# 	plt.plot(loss_vec, 'k-', label='Train Loss')
# 	plt.plot(avg_cost_vec, 'r--', label='Test Loss')
# 	# plt.plot(test_loss, 'b--', label='root squared mean error')

# 	plt.title('Loss (MSE) per Generation')
# 	plt.xlabel('Generation')
# 	plt.ylabel('Loss')
# 	plt.show()

def main():
	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input, state_size
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('dbpedia_rnn.log-mean2mean', False, 'hybrid')
	# X_test, Y_test = adjust_rnn_test_arrays(X_test, Y_test, num_timesteps, state_size)

	# state_size = len(X_train[0])
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default='gru',
                    help='rnn, gru, lstm, or nas')
	parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
	args = parser.parse_args()

	train_neural_network(x, args)

if __name__ == '__main__':
	main()