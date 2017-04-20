import tensorflow as tf
import random as rand
import numpy as np

data = np.reshape(rand.sample(range(10000), 1000), (1000,1))
label = data

#Setting configurations
n_nodes_hl1 = 3 #nodes in hidden layer 1
n_nodes_hl2 = 5 #nodes in hidden layer 2
n_nodes_hl3 = 3 #nodes in hidden layer 3

n_classes = 1 #number of classes = 1. Regression
batch_size = 100

x = tf.placeholder('float', [batch_size, None], name = 'input')
y = tf.placeholder('float') #the size is not specified (it can be anything)

#Defining the computation graph - the neural network model
def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl1])), #randomly (Normal dist) initialized weights of size 784 x n_nodes_hl1
	 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} #randomly initialized weights (Normal distribution) of length n_nodes_hl1  

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
	 'biases':tf.Variable(tf.random_normal([n_classes]))}

	#forward pass
	z1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	a1 = tf.nn.relu(z1)

	z2 = tf.add(tf.matmul(a1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	a2 = tf.nn.relu(z2)

	z3 = tf.add(tf.matmul(a2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	a3 = tf.nn.relu(z3)

	yhat = tf.add(tf.matmul(a3, output_layer['weights']), output_layer['biases'], name = 'output')

	return yhat


#defining the training
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.square(prediction - y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	nEpochs = 100

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(nEpochs):
			epoch_loss = 0
			for batch in range(int(len(data)/batch_size)):
				start = 0 + (batch) * batch_size
				end = 100 + (batch) * batch_size
				epoch_x = data[range(start, end)]
				epoch_y = epoch_x
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', nEpochs, 'loss:', epoch_loss)
		
		sess.run(init)
		save_model(sess)
		error = tf.reduce_mean(tf.square(prediction - y))

		#accuracy = tf.reduce_mean(tf.cast(error, 'float'))
		print('Error:', error)

#saving the trained model
def save_model(session):
	saver = tf.train.Saver()
	tf.global_variables_initializer().run()
	#Saving as Protocol Buffer (pb)
	saver.save(session, '/home/szi/Eclipse/Java/Tensorflow/Tensorflow_Java/Tensorflow_Load/src/resources/identity/model/chkpt', global_step = 0)
	tf.train.write_graph(session.graph.as_graph_def(), '/home/szi/Eclipse/Java/Tensorflow/Tensorflow_Java/Tensorflow_Load/src/resources/identity/model/', 'model.pb', False)
	#Saving as readable file
	#saver.save(session, '/home/szi/Python/SublimeText/Tensorflow/tf_java2/model/trained_model.sd')
	#tf.train.write_graph(session.graph_def, '.', '/home/szi/Python/SublimeText/Tensorflow/tf_java2/model/trained_model.proto', as_text = False)
	#tf.train.write_graph(session.graph_def, '.', '/home/szi/Python/SublimeText/Tensorflow/tf_java2/model/trained_model.txt', as_text = True)
	#Printing node names
	#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
train_neural_network(x)