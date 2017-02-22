
import tensorflow as tf
import re, ast
import numpy as np
from random import sample
import matplotlib.pyplot as plt

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 50

batch_size = 10

# x = tf.placeholder('float',[None,18])
# y = tf.placeholder('float')
x = tf.placeholder(shape=[None, 18], dtype=tf.float32)
y = tf.placeholder('float')

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
	return (m-col_min) / (col_max - col_min)

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

	# x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
	# x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

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
	cost = tf.reduce_mean(tf.abs(y - prediction))
	# cost = tf.sqrt(tf.reduce_mean(y-prediction)/len(x_vals_train))
	# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(1000):
			temp_loss = 0

			rand_index = np.random.choice(len(x_vals_train), size=batch_size)
			rand_x = x_vals_train[rand_index]
			rand_y = np.transpose([y_vals_train[rand_index]])
			# sess.run(optimizer, feed_dict={x: rand_x, y: rand_y})
			_, temp_loss = sess.run([optimizer, cost], feed_dict={x: rand_x, y: rand_y})

			loss_vec.append(temp_loss)

			if (i+1)%100==0:
			    print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

		# evaluate accuracy
		# correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,0))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print "accuracy %.5f'" % sess.run(accuracy, feed_dict={x: x_vals_test, y: y_vals_test})
		
		# prediction=tf.argmax(y,1)
		# print "predictions", prediction.eval(feed_dict={x: x_vals_test}, session=sess)
		# print('Accuracy: %.2f' % accuracy.eval({x:x_vals_test, y: np.transpose([y_vals_test])}))
		# print("{0:.6f}".format(accuracy.eval({x:x_vals_test, y: np.transpose([y_vals_test])})))

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
	print "hi"
	load_data()
	print x
	train_neural_network(x)

if __name__ == '__main__':
	main()