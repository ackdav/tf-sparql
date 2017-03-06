import tensorflow as tf
import re, ast
import numpy as np
from random import sample
import matplotlib.pyplot as plt

n_nodes_hl1 = 80
n_nodes_hl2 = 70
n_nodes_hl3 = 64

x = tf.placeholder(shape=[None, 66], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1],  dtype=tf.float32)

x_vals_train = np.array([])
y_vals_train = np.array([])
x_vals_test = np.array([])
y_vals_test = np.array([])
num_training_samples = 0
batch_size = 10
training_epochs = 200

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
	global num_training_samples

	with open('db-cold-novec-1k.txt-out') as f:
		for line in f:
			line = re.findall(r'\t(.*?)\t', line)
			line = unicode(line[0])
			line = ast.literal_eval(line)
			query_data.append(line)

	y_vals = np.array([x[66] for x in query_data])

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

def neural_net_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([66,n_nodes_hl1])),
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
	sig = tf.sigmoid(prediction)
	cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
	# cost = tf.sqrt(tf.reduce_mean(y-prediction)/len(x_vals_train))
	# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(training_epochs):
			temp_loss = 0
			avg_cost = 0.
			total_batch = int(round(num_training_samples/batch_size))
			for i in range(total_batch-1):
				batch_x = x_vals_train[i*batch_size:(i+1)*batch_size]
				batch_y = y_vals_train[i*batch_size:(i+1)*batch_size]
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x,
    			                                          y: np.transpose([batch_y])})
				loss_vec.append(temp_loss)
				# Compute average loss
				avg_cost += c / total_batch

			# sample prediction
	 		label_value = batch_y
	 		estimate = p
	 		err = label_value-estimate
	 		# print ("num batch:", i)

			# Display logs per epoch step
			if epoch % 50 == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
				print ("[*]----------------------------")
				for i in xrange(3):
					print ("label value:", label_value[i], \
						"estimated value:", estimate[i])
				print ("[*]============================")

		correct_prediction = tf.equal(tf.round(tf.clip_by_value(prediction,0,1)), y)
		correct_prediction = tf.Print(correct_prediction, [correct_prediction])
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print('Accuracy:', accuracy.eval({x: x_vals_test, y: np.transpose([y_vals_test])}))

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