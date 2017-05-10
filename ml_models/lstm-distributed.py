from __future__ import division
import tensorflow as tf
tf.reset_default_graph()

import re, ast, time, os, argparse
import numpy as np
import slurm_manager as slm
from tensorflow.contrib import rnn

from nn_helper import *

LOGDIR = './logs/rnn-dist/'

hm_epochs = 100
batch_size = 16
state_size = 52#66
num_timesteps = 1#4 #How many steps to look back - this is solved with a persistent state
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


cluster, myjob, mytaskid = slm.SlurmClusterManager(num_param_servers=1).build_cluster_spec()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
server = tf.train.Server(cluster,
					job_name=myjob,
					task_index=mytaskid)

def recurrent_neural_network(name="rnn-net"):
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


def train_rnn_dist(benchmark_err, args, log_param, optimizer, batch_size):
	begin_time = time.time()

	if myjob == "ps":
		server.join()
	elif myjob == "worker": 
		with tf.device(tf.train.replica_device_setter(
							worker_device="/job:worker/task:%d" % mytaskid,
							cluster=cluster)):

			prediction, current_state = recurrent_neural_network()
			global_step = tf.Variable(0, name='global_step', trainable=False)

			with tf.name_scope("relativeMeanError"):
				perc_err_train = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err[0]))
				perc_err_test = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err[1]))
				tf.summary.scalar("relativeMeanError train", perc_err_train)
				tf.summary.scalar("relativeMeanError test", perc_err_test)
			with tf.name_scope("optimizer"):
				optimizer_train = tf.train.AdamOptimizer(0.001).minimize(perc_err_train)
				optimizer_test = tf.train.AdamOptimizer(0.001).minimize(perc_err_test)

			init_op = tf.global_variables_initializer()

		if mytaskid==0:
			tf.summary.scalar("relativeMeanError", perc_err_test)
		
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver(sharded=True, reshape=True)
		print("Variables initialized ...", myjob)
		sys.stdout.flush()

		sv = tf.train.Supervisor(is_chief=(mytaskid == 0),
							global_step=global_step,
							summary_op=summary_op,
							logdir=LOGDIR+log_param,
							saver=saver,
							init_op=init_op)
		frequency = 100

		with sv.managed_session(server.target) as sess:
			# if LOGDIR+log_param:
			# 	assert tf.gfile.Exists(LOGDIR+log_param)
			# 	saver.restore(sess, LOGDIR+log_param)
			# 	print('%s: Pre-trained model restored from %s' % (datetime.now(), LOGDIR+log_param))

			writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

			throughputs = []
			throughput_avg = 0.

			results = []

			start_time = time.time()
			for epoch in range(hm_epochs):
				temp_loss = 0.
				avg_cost = 0.
				count = 0
				batch_count = num_training_samples//batch_size
				_current_state = np.zeros((num_layers, 2, batch_size, state_size))

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
					# print ("relative error with model: {0:.3f}".format(perc_err.eval({x: X_test, y: Y_test, init_state: test_state})))
					print ("Error with model: {0:.6f} ### ".format(avg_cost)),
					print ("Error no model: {0:.6f}".format(benchmark_err[1]))
					if epoch != 0:
						results[epoch/2] = avg_cost

				#==========test model============

				#==========training model============
				epoch_start = time.time()
				for i in range(batch_count-1):
					batch_x = X_train[i*batch_size:(i+1)*batch_size]
					batch_y = Y_train[i*batch_size:(i+1)*batch_size]
					
					batch_x = batch_x.reshape([batch_size, num_timesteps, state_size])

					batch_y = np.transpose([batch_y])
					# Run optimization op (backprop) and cost op (to get loss value)
					_, c, p, s_, _current_state, step = sess.run([optimizer_train, perc_err_train, prediction, summary_op, current_state, global_step], 
											feed_dict={x: batch_x, y: batch_y, init_state: _current_state})
					
					# if mytaskid ==0:
					# 	writer.add_summary(s_, epoch)
					# loss_vec.append(c)
				#Throughput queries / second
				epoch_end = time.time()-epoch_start
				throughput = len(X_train) / epoch_end
				throughputs.append(throughput)
				throughput_avg = sum(throughputs) / len(throughputs)	
					# # Compute average loss
					# avg_cost += c / batch_count
					# avg_cost_vec.append(avg_cost)
				# count += 1
				if epoch % 1 == 0:
					print("Job name: %s," % myjob,
							" task: %2d," % mytaskid,
							" Epoch: %2d," % (epoch+1),
							" Cost: %.4f," % c, 
							" AvgThroughput: %3.2fms" % float(throughput_avg))
					sys.stdout.flush()
					#==========training model============
				if epoch % 99 == 0:
					print results
			sess.close()

def make_log_param_string(learning_rate, optimizer, batch_size, warm):
    return "lr_%s_opt_%s_bsize_%s_warm_%s" % (learning_rate, optimizer, batch_size, warm)

# def plot_result(loss_vec, avg_cost_vec):
# 	# Plot loss (MSE) over time
# 	plt.plot(loss_vec, 'k-', label='Train Loss')
# 	plt.plot(avg_cost_vec, 'r--', label='Test Loss')
# 	# plt.plot(test_loss, 'b--', label='root squared mean error')

# 	plt.title('Loss (MSE) per Generation')
# 	plt.xlabel('Generation')
# 	plt.ylabel('Loss')
# 	plt.show()

def main(args):
	print "hi"
	warm = False
	vector_options = {'structure': True,'time': False, 'ged': False,'sim': False,'w2v': False}
	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input, state_size
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('database-iMac.log', warm, vector_options)

	benchmark_err = no_modell_mean_error(Y_train, Y_test)

	start_time=time.clock()
	#setup to find optimal nn
	for optimizer in ['AdadeltaOptimizer']:
		for learning_rate in [0.01]:
			for batch_size in [16]:
				log_param = make_log_param_string(learning_rate, optimizer, batch_size, warm)
				print ('Starting run for %s, optimizer: %s, batch_size: %s, warm: %s' % (log_param, optimizer, batch_size, warm))

				train_rnn_dist(benchmark_err, args, log_param, optimizer, batch_size)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default='nas',
                    help='rnn, gru, lstm, or nas')
	parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layers in the RNN')
	args = parser.parse_args()
	main(args)
