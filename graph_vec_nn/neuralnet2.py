from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import re, ast
from random import sample
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
import numpy as np

# Parameters
learning_rate = 0.02
training_epochs = 500
batch_size = 80
display_step = 1
# Network Parameters
n_hidden_1 = 100 # 1st layer number of features
n_hidden_2 = 80 # 2nd layer number of features
n_hidden_3 = 60
n_hidden_4 = 40

n_classes = 1
X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
# tf Graph input
x = tf.placeholder("float", [None, 62])
y = tf.placeholder("float", [None,1])

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

def load_data():
    query_data = []
    global Y_test
    global X_test
    global Y_train
    global X_train
    global num_training_samples

    with open('db-cold-novec-1k.txt-out') as f:
        for line in f:
            line = re.findall(r'\t(.*?)\t', line)
            line = unicode(line[0])
            line = ast.literal_eval(line)

            # line[-1] = str(line[-1])
            query_data.append(line)

    y_vals = np.array([ float(x[62]) for x in query_data])

    for l_ in query_data:
        del l_[-1]

    x_vals = np.array(query_data)

    # split into test and train
    l = len(x_vals)
    f = int(round(l*0.8))
    indices = sample(range(l), f)

    X_train = x_vals[indices].astype('float32')
    X_test = np.delete(x_vals, indices, 0).astype('float32')

    Y_train = y_vals[indices].astype('float32')
    Y_test = np.delete(y_vals, indices, 0).astype('float32')

    num_training_samples = X_train.shape[0]
    X_train = np.nan_to_num(normalize_cols(X_train))
    X_test = np.nan_to_num(normalize_cols(X_test))

n_input = X_train.shape[0]
load_data()
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([62, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, 1], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([1], 0, 0.1))
}

# Construct model
prediction = multilayer_perceptron(x, weights, biases)
# pred = np.transpose([pred])
# Define loss and optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_training_samples/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            batch_y = np.transpose([batch_y])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = batch_y
        estimate = p
        err = label_value-estimate

        # Display logs per epoch step
        if epoch % 10 == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(10):
                print ("label value:", label_value[i], \
                    "estimated value:", estimate[i])
            print ("[*]============================")

    print ("Optimization Finished!")
    perc_err = tf.divide(tf.abs(\
        tf.subtract(y, prediction)), \
        tf.reduce_mean(y))
    correct_prediction = tf.less(tf.cast(perc_err, "float"), 0.2)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    mean_relative_error = tf.divide(tf.to_float(tf.reduce_sum(perc_err)), Y_test.shape[0])
    Y_test = np.transpose([Y_test])
    print ("Test accuracy: {:.3f}".format(accuracy.eval({x: X_test, y: Y_test})))
    rel_error = mean_relative_error.eval({x: X_test, y: Y_test})
    print ("relative error: ", rel_error)
