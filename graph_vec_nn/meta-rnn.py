import tensorflow as tf
tf.reset_default_graph()

import re, ast, time, argparse, itertools
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from nn_helper import *
import nn

LOGDIR = 'logs/rnn/'

hm_epochs = 100
batch_size = 32
# n_input = 52#66
state_size = 10#66
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

# =============
# Meta-RNN network which optimizes our rnn
# =============
LAYERS = 2
STATE_SIZE = 20
TRAINING_STEPS = 20  # This is 100 in the paper
cell = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in range(LAYERS)])
cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
cell = tf.make_template('cell', cell)

def recurrent_neural_network(x, args, name="rnn-net"):
	with tf.name_scope(name):
		layer = {'weights':tf.Variable(tf.random_normal([state_size,num_classes])),
						'biases':tf.Variable(tf.random_normal([num_classes]))}

		#This unpacks the current state placeholer and assigns it
		l = tf.unstack(init_state, axis=0)
		rnn_tuple_state = tuple( [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)])
		
		def lstm_cell():
			cell = tf.contrib.rnn.NASCell(state_size, reuse=tf.get_variable_scope().reuse)
			return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
			# outputs, current_state = multiPLSTM([cell for _ in range(num_layers)], x, leng, state_size, tuple_state)
		rnn_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple = True)

		outputs, current_state = tf.nn.dynamic_rnn(rnn_cells, x, initial_state=rnn_tuple_state, scope = "layer")

		outputs = tf.transpose(outputs, [1, 0, 2])
		last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
		#outputs gives all states, but we only want last one
		output = tf.matmul(last, layer['weights']) + layer['biases']
		return (output, current_state)

def g_rnn(gradients, state):
    # Make a `batch' of single gradients to create a 
    # "coordinate-wise" RNN as the paper describes. 
    # gradients = tf.expand_dims(gradients, axis=1)
    print type(gradients)

    # gradients = tf.reshape(gradients, [10])
    gradients = tf.expand_dims(gradients, axis=1)
    if state is None:
        state = [[tf.zeros([10,1])] * 2] * num_layers

    update, state = cell(gradients, state)
    # Squeeze to make it a single batch again.
    return tf.squeeze(update, axis=[1]), state

def learn(optimizer, loss, x):
    losses = []
    # x = initial_pos
    state = None
    for _ in range(TRAINING_STEPS):
        # loss = tf.reduce_sum(tf.subtract(y, prediction))
        losses.append(loss)
        grads, = tf.gradients(loss, x)
        update, state = optimizer(grads, state)
        x += update
    return losses

def optimize(loss):
    optimizer = tf.train.AdamOptimizer(0.0001)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.)
    return optimizer.apply_gradients(zip(gradients, v))

def train_neural_network(x, benchmark_err, args, log_param="rnn_meta"):
	global X_test
	prediction, current_state = recurrent_neural_network(x, args)
	begin_time = time.time()
	# cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	with tf.name_scope("relativeMeanError"):
		perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err))
		tf.summary.scalar("relativeMeanError", perc_err)
	# with tf.name_scope("optimizer"):
		# optimizer = tf.train.RMSPropOptimizer(0.005).minimize(perc_err)

	rnn_losses = learn(g_rnn, perc_err, x)
	sum_losses = tf.reduce_sum(rnn_losses)
	apply_update = optimize(sum_losses)

	summary_op = tf.summary.merge_all()
	
	with tf.MonitoredSession() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

		for epoch in range(hm_epochs):
			# temp_loss = 0.
			losses = []
			# avg_cost = 0.
			_current_state = np.zeros((num_layers, 2, batch_size, state_size))
			total_batch = num_training_samples//batch_size

			for i in range(total_batch-1):
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]

				# batch_x = batch_x.reshape([batch_size, num_timesteps, state_size])

				batch_y = np.transpose([batch_y])
				# Run optimization op (backprop) and cost op (to get loss value)
				err, _, c, _current_state = sess.run([sum_losses, apply_update, perc_error, current_state], 
											feed_dict={x: batch_x, y: batch_y, init_state: _current_state})

		print("Total Time: %3.2fs" % float(time.time() - begin_time))


def main():
	warm = False
	vector_options = {'structure': False,'ged': True,'sim': False,'w2v': False}
	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input, state_size
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('database.log-complete', warm, vector_options)
	benchmark_err = no_modell_mean_error(Y_train, Y_test)
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default='lstm',
                    help='rnn, gru, lstm, or nas')
	parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
	args = parser.parse_args()

	train_neural_network(x,benchmark_err, args)

if __name__ == '__main__':
	main()