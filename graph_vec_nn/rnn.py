import tensorflow as tf
tf.reset_default_graph()

import re, ast, time, argparse, itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from nn_helper import *

hm_epochs = 250
batch_size = 256
n_input = 62
input_dimension = 62 
sequence_length = 3 #How many steps to look back

x = tf.placeholder(shape=[None, sequence_length, input_dimension], dtype=tf.float32, name="x")
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
y_vals = np.array([])

# Training loop
loss_vec = []
test_loss = []
avg_cost_vec = []

def recurrent_neural_network(x, args):
	layer = {'weights':tf.Variable(tf.random_normal([n_input*4,1])),
						'biases':tf.Variable(tf.random_normal([1]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, input_dimension])
	x = tf.split(x, sequence_length, 0)
	print (x)
	if args.model == 'rnn':
		cell_fn = rnn.BasicRNNCell
	elif args.model == 'gru':
		cell_fn = rnn.GRUCell
	elif args.model == 'lstm':
		cell_fn = rnn.BasicLSTMCell
	elif args.model == 'nas':
		cell_fn = rnn.NASCell
	else:
		raise Exception("model type not supported: {}".format(args.model))

	cells = []
	for _ in xrange(args.num_layers):
		cell = cell_fn(n_input*4)
		cells.append(cell)

	rnn_cells = tf.contrib.rnn.MultiRNNCell(cells)
	outputs, states = tf.contrib.rnn.static_rnn(rnn_cells, x, dtype = tf.float32)

	#outputs gives all states, but we only want last one
	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	return output


def train_neural_network(x, args):
	prediction = recurrent_neural_network(x, args)
	begin_time = time.time()
	# cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), tf.reduce_mean(y)))
	optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(perc_err)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			temp_loss = 0.
			avg_cost = 0.

			total_batch = int(round(num_training_samples/batch_size))

			for i in range(total_batch-1):
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]
				#reshape to (batch_size, sequence_length / element )
				enlarged_batch = []
				for id, obj in enumerate(batch_x):
					try:
						obj = np.append(obj, batch_x[id-1]) #TODO: Correct this to work to match epoch start
						obj = np.append(obj, batch_x[id-2])
						# obj = np.append(obj, batch_x[id-3])
						enlarged_batch.append(obj)
					except:
						print "shit" + str(id)
				batch_x = np.array([enlarged_batch])

				batch_x = batch_x.reshape([batch_size, sequence_length, input_dimension])
				batch_y = np.transpose([batch_y])
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c, p = sess.run([optimizer, perc_err, prediction], feed_dict={x: batch_x,
    			                                          y: batch_y})
				loss_vec.append(c)
				# Compute average loss
				avg_cost += c / total_batch
				avg_cost_vec.append(avg_cost)

			# sample prediction
	 		label_value = batch_y
	 		estimate = p
	 		err = label_value-estimate
	 		# print ("num batch:", i)
	 		# perc_err = tf.reduce_mean((batch_y-p)/batch_y)
			# Display logs per epoch step
			if epoch % 5 == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
				print ("[*]----------------------------")
				for i in xrange(4):
					print ("label value:", label_value[i], \
						"estimated value:", estimate[i])
				print ("[*]============================")

		# X_test = X_test.reshape((-1, sequence_length, input_dimension))

		# print ("RMSE: {:.3f}".format(cost.eval({x: X_test.reshape((-1, sequence_length, input_dimension)), y: Y_test})))
		print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test.reshape((-1, sequence_length, input_dimension)), y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(Y_train, Y_test)))
		print("Total Time: %3.2fs" % float(time.time() - begin_time))


def plot_result(loss_vec, avg_cost_vec):
	# Plot loss (MSE) over time
	plt.plot(loss_vec, 'k-', label='Train Loss')
	plt.plot(avg_cost_vec, 'r--', label='Test Loss')
	# plt.plot(test_loss, 'b--', label='root squared mean error')

	plt.title('Loss (MSE) per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()

def main():
	print "hi"
	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('random200k.log-result', False, 'hybrid')
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default='nas',
                    help='rnn, gru, lstm, or nas')
	parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layers in the RNN')
	args = parser.parse_args()

	train_neural_network(x, args)

if __name__ == '__main__':
	main()