
import tensorflow as tf
import re, ast
import numpy as np
from random import sample
import matplotlib.pyplot as plt


# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

batch_size = 100

# x = tf.placeholder('float',[None,784])
# y = tf.placeholder('float')
x_data = tf.placeholder(shape=[None, 18], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

x_vals_train = np.array([])
y_vals_train = np.array([])
x_vals_test = np.array([])
y_vals_test = np.array([])

# Training loop
loss_vec = []
test_loss = []

# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
	col_max = m.max(axis=0)
	col_min = m.min(axis=0)
		# if (col_max - col_min) > 0:
	return (m-col_min) / (col_max - col_min)
	# else:
	# 	return 0

def load_data():
	query_data = []
	global y_vals_test
	global x_vals_test
	global y_vals_train
	global x_vals_train

	with open('tf-db-cold.txt') as f:
		for line in f:
			line = re.findall(r'\t(.*?)\t', line)
			line = unicode(line[0])
			line = ast.literal_eval(line)
			query_data.append(line)

	y_vals = np.array([x[18] for x in query_data])

	for list in query_data:
		del list[-1]

	x_vals = np.array(query_data)

	# split into test and train 
	l = len(x_vals)
	f = int(round(l*0.8))
	indices = sample(range(l), f)
	x_vals_train = x_vals[indices]
	x_vals_test = np.delete(x_vals, indices, 0)
	# print x_vals_test.shape
	y_vals_train = y_vals[indices]
	y_vals_test = np.delete(y_vals, indices, 0)

	x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
	x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


def neural_net_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([18,n_nodes_hl1])),
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
	# # Declare loss function (L1)
	cost = tf.reduce_mean(tf.abs(y_target - prediction))
	# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y_target))
	optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(1000):
			rand_index = np.random.choice(len(x_vals_train), size=batch_size)
			rand_x = x_vals_train[rand_index]
			rand_y = np.transpose([y_vals_train[rand_index]])
			sess.run(optimizer, feed_dict={x_data: rand_x, y_target: rand_y})

			temp_loss = sess.run(cost, feed_dict={x_data: rand_x, y_target: rand_y})
			loss_vec.append(temp_loss)

			test_temp_loss = sess.run(cost, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
			test_loss.append(test_temp_loss)
			
			if (i+1)%100==0:
			    print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

		# evaluate accuracy
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_target,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x_data:x_vals_test, y_target: np.transpose([y_vals_test])}))

		plot_result(loss_vec, test_loss)

def plot_result(loss_vec, test_loss):
	# Plot loss (MSE) over time
	plt.plot(loss_vec, 'k-', label='Train Loss')
	plt.plot(test_loss, 'r--', label='Test Loss')
	plt.title('Loss (MSE) per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()

def main():
	# train_neural_network(x)
	print "hi"
	load_data()
	train_neural_network(x_data)

if __name__ == '__main__':
	main()