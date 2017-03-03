from __future__ import print_function
import re, ast
import numpy as np
import tensorflow as tf
from random import sample

# tf Graph Input
X = tf.placeholder("float", [None, 34])
Y = tf.placeholder("float", [None, 1])
X_test = tf.placeholder("float", [34])

K = 4

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

    with open('verycomp.txt') as f:
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


def knn_model(x):
    # Euclidean Distance
    distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_vals_train, X_test)), reduction_indices=1)))
    # Prediction: Get min distance neighbors
    values, indices = tf.nn.top_k(distance, k=K, sorted=False)
    nearest_neighbors = []
    for i in range(K):
        nearest_neighbors.append(tf.argmax(y_vals_train[indices[i]], 0))

    neighbors_tensor = tf.stack(nearest_neighbors)
    y, idx, count = tf.unique_with_counts(neighbors_tensor)
    pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
    return pred

def train_knn(x):
    pred = knn_model(x)
    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        accuracy = 0.

        # loop over test data
        for i in range(len(x_vals_test)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={X: x_vals_train, Y: y_vals_train, X_test: x_vals_test[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", nn_index,
                 "True Class:", np.argmax(Yte[i]))
            #Calculate accuracy
            if nn_index == np.argmax(Yte[i]):
                accuracy += 1. / len(x_vals_test)
        print("Done!")
        print("Accuracy:", accuracy)

def main():
    x = load_data()
    train_knn(x)

if __name__ == '__main__':
    main()