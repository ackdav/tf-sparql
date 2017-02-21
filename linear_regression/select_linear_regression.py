'''
following: https://github.com/jorditorresBCN/LibroTensorFlow <- doesn't work due to breaking changes

This scripts loads already extracted data from "bio_select_variables" and finds linear regression between number of select variables and execution time
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from bio_select_variables import *
rng = np.random

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12


# Parameters
learning_rate = 0.01
epochs = 1000

#Set training arrays
train_X = np.array([])
train_Y = np.array([])

# Set model weights
W = tf.Variable(rng.randn(), name="weight", dtype=tf.float64)
b = tf.Variable(rng.randn(), name="bias", dtype=tf.float64)

n_samples = 0

#load BioPortal data and append training vectors
def load_bio_data():
    db = readout_feature()

    global train_X
    global train_Y
    global n_samples

    #load in first 2000 datapoints
    for entry in (line for i, line in enumerate(db) if i<=2000):
        entry_split = re.split(r'[\t|\n]', entry)

        try:
            #first try at taking out outliers, plot is unreadable otherwise
            #TODO - find better solution
            if 0 < float(entry_split[1]) and float(entry_split[1]) < 10 and float(entry_split[0])<7:
                # print(entry)
                train_X = np.append(train_X,float(entry_split[0]))
                train_Y = np.append(train_Y,float(entry_split[1]))
                n_samples += 1
        except ValueError,e:
            print ("error",e,"on line",entry_split)

def load_dbpedia_data():

    global train_X
    global train_Y
    global n_samples

    # open(os.path.dirname(__file__) + '/../data.yml')
    with open(os.path.dirname(__file__) + '/../data/dbpedia/dbpedia-20k-cold.txt', 'r+') as f:
        for line in f:
            
            #split db to get query
            line_split = re.split(r'[\t]', line)

            #split to get only part between where and select            
            # select_where = re.findall('SELECT(.*?)WHERE', line_split[0], re.DOTALL)
            
            union_count = line_split[0].count("UNION")

            # if len(select_where) > 0:
            #     select_variables = select_where[0].count("?")


            try:
                if union_count>=0:
                    train_X = np.append(train_X,float(union_count))
                    train_Y = np.append(train_Y,float(line_split[1]))
                    n_samples += 1

            except ValueError,e:
                print ("error",e,"on line",line_split)

def linear_model():
    # Construct a linear model
    return tf.add(tf.mul(train_X, W), b)

def train_linear_model():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        pred = linear_model()

        # # Mean squared error
        # cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*len(data))
        cost = tf.reduce_mean(tf.square(pred - train_Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(cost)


        for epoch in range(epochs):
            sess.run(train)
            print(epoch, sess.run(W), sess.run(b))
            print(epoch, sess.run(cost))

        #Graphic display
        plt.plot(train_X, train_Y, 'ro')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), linestyle="-")

        plt.xlabel('SELECT variables')
        plt.ylabel('Execution time')
        plt.xticks([1,2,3,4])
        # plt.xlim(0.97,4.03)
        # plt.ylim(-0.03,8.1)
        plt.legend()
        plt.show()

def main():
    load_dbpedia_data()
    train_linear_model()

if __name__ == '__main__':
    main()