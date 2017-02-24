import tensorflow as tf
import re, ast
import numpy as np
from random import sample
import matplotlib.pyplot as plt
# from tensorflow.python.ops import rnn, rnn_cell


hm_epochs = 100
batch_size = 64
chunk_size = 9
n_chunks = 2

rnn_size = 64

x = tf.placeholder(shape=[None, n_chunks, chunk_size], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1],  dtype=tf.float32)

x_vals_train = np.array([])
y_vals_train = np.array([])
x_vals_test = np.array([])
y_vals_test = np.array([])


# Training loop
loss_vec = []
test_loss = []
avg_cost_vec = []

# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
	col_max = m.max(axis=0)
	col_min = m.min(axis=0)
	return (m-col_min) / (col_max - col_min)

def load_data():
	query_data = []
	global y_vals_test
	global x_vals_test
	global y_vals_train
	global x_vals_train
	global num_training_samples

	with open('tf-db-cold.txt') as f:
		for line in f:
			line = re.findall(r'\t(.*?)\t', line)
			line = unicode(line[0])
			line = ast.literal_eval(line)
			query_data.append(line)

	y_vals = np.array([x[18]*10000 for x in query_data])

	for list in query_data:
		del list[-1]

	x_vals = np.array(query_data)

	# split into test and train 
	l = len(x_vals)
	f = int(round(l*0.8))
	indices = sample(range(l), f)
	x_vals_train = x_vals[indices]
	x_vals_test = np.delete(x_vals, indices, 0)

	y_vals_train = y_vals[indices]
	y_vals_test = np.delete(y_vals, indices, 0)

	num_training_samples = x_vals_train.shape[0]
	# x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
	# x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

def recurrent_neural_network(x):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,1])),
						'biases':tf.Variable(tf.random_normal([1]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype = tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output


def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	sig = tf.sigmoid(prediction)
	cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	# cost = tf.sqrt(tf.reduce_mean(y-prediction)/len(x_vals_train))
	# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			temp_loss = 0.
			avg_cost = 0.

			total_batch = int(round(num_training_samples/batch_size))
			for i in range(total_batch-1):
				batch_x = x_vals_train[i*batch_size:(i+1)*batch_size]
				batch_y = y_vals_train[i*batch_size:(i+1)*batch_size]
				batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

				# Run optimization op (backprop) and cost op (to get loss value)
				_, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x,
    			                                          y: np.transpose([batch_y])})
				loss_vec.append(c)
				# Compute average loss
				avg_cost += c / total_batch
				avg_cost_vec.append(avg_cost)

			# sample prediction
	 		label_value = batch_y
	 		estimate = p
	 		err = label_value-estimate
	 		# print ("num batch:", i)
	 		perc_err = tf.reduce_mean((batch_y-p)/batch_y)
			# Display logs per epoch step
			if epoch % 50 == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
				print ("[*]----------------------------")
				for i in xrange(4):
					print ("label value:", label_value[i], \
						"estimated value:", estimate[i])
				print ("[*]============================")

		perc_err = tf.divide(tf.abs(tf.subtract(y_vals_test, tf.cast(prediction, dtype=tf.float64))), tf.reduce_mean(y_vals_test))
		correct_prediction = tf.less(tf.cast(perc_err, "float"), 0.15)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		mean_relative_error = tf.divide(tf.to_float(tf.reduce_sum(perc_err)), y_vals_test.shape[0])

		print "Test accuracy: {:.3f}".format(accuracy.eval({x:x_vals_test.reshape((-1, n_chunks, chunk_size)), y: np.transpose([y_vals_test])}))
		print "relative error: ",mean_relative_error.eval({x:x_vals_test.reshape((-1, n_chunks, chunk_size)), y: np.transpose([y_vals_test])})
		plot_result(loss_vec, avg_cost_vec)

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
	load_data()
	train_neural_network(x)

if __name__ == '__main__':
	main()