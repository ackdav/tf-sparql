import tensorflow as tf
import re, ast, sys
import numpy as np
from random import sample
# import matplotlib.pyplot as plt

n_nodes_hl1 = 0
n_nodes_hl2 = 0
n_nodes_hl3 = 0

x = tf.placeholder(shape=[None, 34], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1],  dtype=tf.float32)

x_vals_train = np.array([], dtype='float32')
y_vals_train = np.array([], dtype='float32')
x_vals_test = np.array([], dtype='float32')
y_vals_test = np.array([], dtype='float32')
num_training_samples = 0
batch_size = 30
training_epochs = 350

# Training loop
loss_vec = []
test_loss = []
avg_cost_vec = []

def setting_nodes(l1=50, l2=40, l3=30):
	global n_nodes_hl1
	global n_nodes_hl2
	global n_nodes_hl3
	n_nodes_hl1 = l1
	n_nodes_hl2 = l2
	n_nodes_hl3 = l3

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
			# line[-1] = str(line[-1])
			query_data.append(line)

	y_vals = np.array([ float(x[34]) for x in query_data])

	for l_ in query_data:
		del l_[-1]

	x_vals = np.array(query_data)

	# split into test and train 
	l = len(x_vals)
	f = int(round(l*0.8))
	indices = sample(range(l), f)
	x_vals_train = x_vals[indices].astype('float32')
	x_vals_test = np.delete(x_vals, indices, 0).astype('float32')

	y_vals_train = y_vals[indices].astype('float32')
	y_vals_test = np.delete(y_vals, indices, 0).astype('float32')

	num_training_samples = x_vals_train.shape[0]
	x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
	x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

def neural_net_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([34,n_nodes_hl1])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,1])),
		'biases':tf.Variable(tf.random_normal([1]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_net_model(x)
	cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	# cost = tf.sqrt(tf.reduce_mean(y-prediction)/len(x_vals_train))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(training_epochs):
			temp_loss = 0.
			avg_cost = 0.

			total_batch = int(round(num_training_samples/batch_size))
			for i in range(total_batch-1):
				batch_x = x_vals_train[i*batch_size:(i+1)*batch_size]
				batch_y = y_vals_train[i*batch_size:(i+1)*batch_size]

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
			# if epoch % 50 == 0:
			# 	print ("Epoch:", '%04d' % (epoch+1), "cost=", \
			# 		"{:.9f}".format(avg_cost))
			# 	print ("[*]----------------------------")
			# 	for i in xrange(4):
			# 		print ("label value:", label_value[i], \
			# 			"estimated value:", estimate[i])
			# 	print ("[*]============================")
		perc_err = tf.divide(tf.abs(\
			tf.subtract(y, prediction)), \
			tf.reduce_mean(y))
		correct_prediction = tf.less(tf.cast(perc_err, "float"), 0.2)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		mean_relative_error = tf.divide(tf.to_float(tf.reduce_sum(perc_err)), y_vals_test.shape[0])

		print "Test accuracy: {:.3f}".format(accuracy.eval({x: x_vals_test, y: np.transpose([y_vals_test])}))
		# _, c, p = sess.run([optimizer, mean_relative_error, prediction], feed_dict={x: x_vals_test, y: np.transpose([y_vals_test])})
		# print _, c, p 
		rel_error = mean_relative_error.eval({x: x_vals_test, y: np.transpose([y_vals_test])})
		print "relative error: ", rel_error
		return rel_error
		# plot_result(loss_vec, avg_cost_vec)

# def plot_result(loss_vec, avg_cost_vec):
# 	# Plot loss (MSE) over time
# 	plt.plot(loss_vec, 'k-', label='Train Loss')
# 	plt.plot(avg_cost_vec, 'r--', label='Test Loss')
# 	plt.title('Loss (MSE) per Generation')
# 	plt.xlabel('Generation')
# 	plt.ylabel('Loss')
# 	plt.show()

def testscript():
	setting_nodes()
	load_data()
	train_neural_network(x)

def main():
	print "hi"
	results = []
	# for i in range(0, 30, 3):
	# 	for j in range(0, 30, 3):
	# 		for k in range(0, 30, 3):
	# 			setting_nodes(i, j, k)
	# 			load_data()
	# 			rel_error = train_neural_network(x)
	# 			sys.stdout.flush()
	# 			# print (str(i) + '\t' + str(j) + '\t' + str(k) + '\t' + str(rel_error))
	# 			results.append(str(i) + '\t' + str(j) + '\t' + str(k) + '\t' + str(rel_error))
	
	load_data()
	train_neural_network(x)
	
	# with open('res', 'a') as out:
	# 	for _ in results:
	# 		out.write(_)

if __name__ == '__main__':
	main()