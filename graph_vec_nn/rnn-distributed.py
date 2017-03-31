import tensorflow as tf
tf.reset_default_graph()

import re, ast, time, os, argparse
import numpy as np
import slurm_manager as slm
from tensorflow.contrib import rnn

from nn_helper import *

LOGDIR = './logs/rnn-dist/'

hm_epochs = 250
batch_size = 64
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

if os.environ['USER']=='ackdav':
	cluster, myjob, mytaskid = slm.SlurmClusterManager().build_cluster_spec()
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
	server = tf.train.Server(cluster,
					job_name=myjob,
					task_index=mytaskid)

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


def train_neural_network(args, learning_rate, log_param, optimizer, batch_size):
	begin_time = time.time()

	if myjob == "ps":
		server.join()
	elif myjob == "worker": 
		with tf.device(tf.train.replica_device_setter(
				worker_device="/job:worker/task:%d" % mytaskid,
				cluster=cluster)):
			prediction = recurrent_neural_network(x, args)
			global_step = tf.Variable(0, name='global_step', trainable=False)

			with tf.name_scope("RMSE"):
				cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))), name="RMSE")
				tf.summary.scalar("RMSE", cost)
			with tf.name_scope("relativeMeanError"):
				perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), tf.reduce_mean(y)))
				tf.summary.scalar("relativeMeanError", perc_err)

			with tf.name_scope("train"):
				if optimizer == 'AdagradOptimizer':
					optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(perc_err, global_step=global_step)
				if optimizer == 'FtrlOptimizer':
					optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(perc_err, global_step=global_step)
				if optimizer == 'AdadeltaOptimizer':
					optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(perc_err, global_step=global_step)
				if optimizer == 'AdamOptimizer':
					optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(perc_err, global_step=global_step)

			summary_op = tf.summary.merge_all()

			init_op = tf.global_variables_initializer()
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
			# saver.restore(sess, LOGDIR + os.path.join(log_param, "model.ckpt"))
			start_time = time.time()
			for epoch in range(hm_epochs):
				temp_loss = 0.
				avg_cost = 0.
				count = 0
				batch_count = int(round(num_training_samples/batch_size))

				for i in range(batch_count-1):
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
					_, c, p, step = sess.run([optimizer, perc_err, prediction, global_step], feed_dict={x: batch_x, y: batch_y})


					# loss_vec.append(c)
					# # Compute average loss
					# avg_cost += c / batch_count
					# avg_cost_vec.append(avg_cost)
					count += 1
					if count % frequency == 0 or i+1 == batch_count:
						elapsed_time = time.time() - start_time
						start_time = time.time()
						print("Count: %d," % (step+1), 
									" Epoch: %2d," % (epoch+1), 
									" Batch: %3d of %3d," % (i+1, batch_count), 
									" Cost: %.4f," % c, 
									" AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
						sys.stdout.flush()
						count = 0
					if mytaskid==0 and epoch % 50 == 0:
						saver.save(sess, LOGDIR + os.path.join(log_param, "model.ckpt"))

			# # sample prediction
		# 	label_value = batch_y
		# 	estimate = p
		# 	err = label_value-estimate
		# 	# print ("num batch:", i)
		# 	perc_err = tf.reduce_mean((batch_y-p)/batch_y)
			# # Display logs per epoch step
			# if epoch % 50 == 0:
			# 	print ("Epoch:", '%04d' % (epoch+1), "cost=", \
			# 		"{:.9f}".format(avg_cost))
			# 	print ("[*]----------------------------")
			# 	for i in xrange(4):
			# 		print ("label value:", label_value[i], \
			# 			"estimated value:", estimate[i])
			# 	print ("[*]============================")

			print ("RMSE: {:.3f}".format(cost.eval({x: X_test.reshape((-1, sequence_length, input_dimension)), y: Y_test})))
			print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test.reshape((-1, sequence_length, input_dimension)), y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(Y_train, Y_test)))
			print("Total Time: %3.2fs" % float(time.time() - begin_time))
		sv.stop()
		# sess.close()

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
	global X_train, X_test, Y_train, Y_test, num_training_samples, n_input
	X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('random200k.log-result', warm, 'hybrid')
	start_time=time.clock()
	#setup to find optimal nn
	for optimizer in ['AdadeltaOptimizer']:
		for learning_rate in [0.01]:
			for batch_size in [64]:
				log_param = make_log_param_string(learning_rate, optimizer, batch_size, warm)
				print ('Starting run for %s, optimizer: %s, batch_size: %s, warm: %s' % (log_param, optimizer, batch_size, warm))

				train_neural_network(args, learning_rate, log_param, optimizer, batch_size)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type=str, default='nas',
                    help='rnn, gru, lstm, or nas')
	parser.add_argument('--num_layers', type=int, default=4,
                    help='number of layers in the RNN')
	args = parser.parse_args()
	main(args)
