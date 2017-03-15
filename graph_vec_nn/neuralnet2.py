from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.reset_default_graph()

from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import re, ast, os
from random import sample
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
import numpy as np

LOGDIR = '/tmp/'

# Parameters
learning_rate = 0.005
training_epochs = 600
batch_size = 150
display_step = 1
# Network Parameters
n_hidden_1 = 350 # 1st layer number of features
n_hidden_2 = 300 # 2nd layer number of features
n_hidden_3 = 250
n_hidden_4 = 170

n_classes = 1
X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
y_vals = np.array([])
no_model_error = 0.
# tf Graph input
x = tf.placeholder("float", [None, 78], name="x")
y = tf.placeholder("float", [None,1], name="y")

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
    global y_vals
    global num_training_samples

    volume = 0.
    with open('log160k.log-out-full') as f:
        for line in f:
            line1 = re.findall(r'\t(.*?)\t', line)
            volume = line.split('\t')
            volume = volume[2].strip('\n')
            # print (volume)
            line1 = unicode(line1[0])
            line1 = ast.literal_eval(line1)
            # line[-1] = str(line[-1])
            # line1 = [int(volume)] + line1
            # print (line1)
            query_data.append(line1)

    y_vals = np.array([ float(x[78]) for x in query_data])

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

def no_modell_mean_error(y_vals):
    mean_ = y_vals.mean()
    mean_sum = 0.
    for y_ in y_vals:
        mean_error = abs(y_- mean_) / mean_
        mean_sum += mean_error
    return mean_sum / y_vals.shape[0]

n_input = X_train.shape[0]
load_data()
# Create model

def multilayer_perceptron(x, weights, biases, name="neuralnet"):
    with tf.name_scope(name):
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
    'h1': tf.Variable(tf.random_normal([78, n_hidden_1], 0, 0.1), name="h1W"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1), name="h2W"),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1), name="h3W"),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1), name="h4W"),
    'out': tf.Variable(tf.random_normal([n_hidden_4, 1], 0, 0.1), name="hOutW")
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1), name="h1W"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1), name="h2W"),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1), name="h3W"),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1), name="h4W"),
    'out': tf.Variable(tf.random_normal([1], 0, 0.1), name="hOutW")
}

# Construct model
prediction = multilayer_perceptron(x, weights, biases)
# pred = np.transpose([pred])
# Define loss and optimizer
with tf.name_scope("RMSE"):
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))), name="RMSE")
    tf.summary.scalar("RMSE", cost)

with tf.name_scope("train"):
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

summ = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    #TensorBoard 
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
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

            if i % 10 == 0:
                [train_accuracy, s] = sess.run([cost, summ], feed_dict={x: batch_x, y: batch_y})
                writer.add_summary(s, i)


        # sample prediction
        label_value = batch_y
        estimate = p
        err = label_value-estimate

        # Display logs per epoch step
        if epoch % 50 == 0:
            # sess.run(assignment, feed_dict={x: X_test, y: Y_test})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), epoch)
            tf.summary.scalar("perc_error", perc_error)


            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(10):
                print ("label value:", label_value[i], \
                    "estimated value:", estimate[i])
            print ("[*]============================")

    print ("Optimization Finished!")
    with tf.name_scope("relativeMeanError"):
        perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), tf.reduce_mean(y)))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))

        # mean_relative_error = tf.divide(tf.to_float(tf.reduce_sum(perc_err)), Y_test.shape[0])
        Y_test = np.transpose([Y_test])
        print ("RMSE: {:.3f}".format(rmse.eval({x: X_test, y: Y_test})))
        print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test, y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(y_vals)))

        tf.summary.scalar("relative_mean_error", perc_err)

# def main():


# if __name__ == '__main__':
#     main()
