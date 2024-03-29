from __future__ import division
import tensorflow as tf
tf.reset_default_graph()

from tensorflow.python.client import timeline #profiling
import re
import ast
import os
import sys
import time
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from heapq import nlargest

# =======plotting settings=======
sns.set()
sns.set_style("darkgrid")
sns.set_color_codes("dark")

#db reading helper
from nn_helper import *

LOGDIR = 'logs/neuralnet2/'

# Parameters
training_epochs = 80

n_classes = 1
n_input = 52 
X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
no_model_error = 0.

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None,1], name="y")

def multilayer_perceptron(x, layer_config, name="neuralnet"):
    '''
    sets up neural network in a dynamic way to test effectiveness of height and depth of network
    code idea from: https://pythonprogramming.net/community/262/TensorFlow%20For%20loop%20to%20set%20weights%20and%20biases/
    (original code has errors)

    Args:
        x: query vector of batch_size length
        layer_config: config for layer sizes, e.g. [n_input, 1024, n_classes] has 1 hidden layer of size 1024

    Returns:
        last layer: layer of size 'n_classes' (1 in our case)
    '''
    layers = {}
    layers_compute = {}

    with tf.name_scope(name):
        for i in range(1, len(layer_config)):
            # new_layer = {'weights': tf.Variable(tf.random_normal(shape=[layer_config[i-1], layer_config[i]], mean=0, stddev=1)),
            #             'biases': tf.Variable(tf.random_normal(shape=[layer_config[i]], mean=0, stddev=1))}
            new_layer = {'weights': tf.get_variable(name="w"+str(i), shape=[layer_config[i-1], layer_config[i]], initializer=tf.contrib.layers.xavier_initializer()),
                        'biases': tf.get_variable(name="b"+str(i), shape=[layer_config[i]], initializer=tf.contrib.layers.xavier_initializer())}
            
            layers[i-1] = new_layer

            with tf.name_scope("weights"):
                tf.summary.histogram("w_l"+str(i)+"_summary", new_layer['weights'])

            with tf.name_scope("biases"):
                tf.summary.histogram("b_l"+str(i)+"_summary", new_layer['biases'])

            l = tf.add(tf.matmul(x if i == 1 else layers_compute[i-2], layers[i-1]['weights']), layers[i-1]['biases'])
            
            with tf.name_scope(name):
                l = tf.nn.relu(l) if i != len(layer_config)-1 else l
                l = tf.nn.dropout(l, keep_rate) if i != len(layer_config)-1 else l

            layers_compute[i-1] = l

    lastlayer = len(layers_compute)-1
    return layers_compute[lastlayer]

def run_nn_model(learning_rate, benchmark_err, log_param, optimizer, batch_size, layer_config, gl):
    '''Builds session and executes model in run_nn_model. prints results and plots results.

    Args:
        learning_rate: 'float', learning rate of optimizer
        benchmark_err: 'tuple', error of dataset (without any model) of training and testset
        log_param: 'string', settings string for tensorboard saving - keeps logs separated
        optimizer: 'string', identifies correct optimizer to load
        batch_size: 'int', batch size for model
        layer_config: 'list', config for rnn layers - see run_nn_model()
    '''
    begin_time = time.time()
    prediction = multilayer_perceptron(x, layer_config)

    # ========= profiling ========
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # ========= profiling ========

    with tf.name_scope("relativeMeanError"):
        perc_err_train = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err[0]))
        perc_err_test = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), benchmark_err[1]))
        tf.summary.scalar("relativeMeanError train", perc_err_train)
        tf.summary.scalar("relativeMeanError test", perc_err_test)

    with tf.name_scope("optimizer"):
        if optimizer == 'AdagradOptimizer':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(perc_err)
        if optimizer == 'FtrlOptimizer':
            optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(perc_err)
        if optimizer == 'AdadeltaOptimizer':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(perc_err)
        if optimizer == 'AdamOptimizer':
            optimizer_train = tf.train.AdamOptimizer(learning_rate).minimize(perc_err_train)
            optimizer_test = tf.train.AdamOptimizer(learning_rate).minimize(perc_err_test)
        if optimizer == 'RMSPropOptimizer':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(perc_err)
    # merge all summaries into a single "operation" which we can execute in a session 
    summary_op = tf.summary.merge_all()
    
    # Launch the graph
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

        test_err = []
        train_batch_loss_y = []
        train_batch_loss_x = []
        last_n_results = []
        results = []
        final_res = 0.
        e_opt = 0.
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(num_training_samples/batch_size)
            # Loop over all batches


            # ========== training ==========
            for i in range(total_batch-1):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = Y_train[i*batch_size:(i+1)*batch_size]
                batch_y = np.transpose([batch_y])
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c_p, p, s = sess.run([optimizer_train, perc_err_train, prediction, summary_op], feed_dict={x: batch_x, y: batch_y})

                # ========= profiling ======== , options=run_options, run_metadata=run_metadata
                # tl = timeline.Timeline(run_metadata.step_stats)
                # ctf = tl.generate_chrome_trace_format()
                # with open('timeline.json', 'w+') as f:
                #     f.write(ctf)
                # ========= profiling ========
                avg_cost += c_p / total_batch
                if i % 150 == 0:
                    train_batch_loss_y.append(c_p)
                    train_batch_loss_x.append(epoch + i/total_batch)

            label_value = batch_y
            estimate = p
            err = label_value-estimate

            # if epoch % 1 == 0:
            #     # sess.run(assignment, feed_dict={x: X_test, y: Y_test})
            #     writer.add_summary(s, epoch)

            #     print ("Epoch:", '%04d' % (epoch+1), "cost=", \
            #         "{:.9f}".format(avg_cost))
            #     print ("[*]----------------------------")
            #     for i in xrange(4):
            #         print ("label value:", label_value[i], \
            #             "estimated value:", estimate[i])
            #     print ("[*]============================")
            #     sys.stdout.flush()
            # ========== training ==========


            # ========== test ==========
            if epoch % 2 == 0:
                # mean_relative_error = tf.divide(tf.to_float(tf.reduce_sum(perc_err)), Y_test.shape[0])
                # print ("RMSE: {:.3f}".format(cost.eval({x: X_test, y: Y_test})))

                # Y_test = np.transpose([Y_test])
                _, c_p, p, s_ = sess.run([optimizer_test, perc_err_test, prediction, summary_op], feed_dict={x: X_test, y: Y_test})
                
                if epoch == 0:
                    e_opt = c_p
                early_stop = 100 * ((c_p/e_opt)-1)
                if early_stop > gl and epoch > 10:
                    break
                if c_p < e_opt:
                    e_opt = c_p

                    
                #calculate mean over last 5 results
                if len(last_n_results) > 4:
                    del last_n_results[-1]
                last_n_results.insert(0, c_p)

                results.append(c_p)
                end_res = sum(last_n_results) / len(last_n_results)
                final_res = end_res


            
                


                # stops.append(early_stop)

                
                # print ("epoch: {:.1f}".format(epoch), "last error: {:.5f}".format(c_p), "avg last 5: {:.5f}".format(end_res), "without model: {:.3f}".format(benchmark_err[1]))

                test_err.append(c_p)
                writer.add_summary(s_, epoch)

                # if epoch > 9:
                #     label_value = Y_test
                #     #find k largest error
                #     k_values = nlargest(5, err)
                #     ids_ = []
                #     for idx, _ in enumerate(err):
                #         if _ in k_values:
                #             ids_.append(idx)
                #     for idx_ in ids_:
                #         print label_value[idx_]
            # ========== test ==========
        return final_res

            # if epoch % 99 == 0 and epoch != 0: 
            #     plot_res(test_err, (train_batch_loss_x, train_batch_loss_y), benchmark_err, epoch)
                # saver.save(sess, LOGDIR + os.path.join(log_param, "model.ckpt"), epoch)

            # if epoch % 99 == 0 and epoch != 0:
            #     with open('results-nn.txt', 'a+') as out_:
            #         out_.write('cold-structure' + str(results)+ '\n')

        # ========= profiling ========
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)
        # ========= profiling ========
        # print ("RMSE: {:.3f}".format(cost.eval({x: X_test, y: Y_test})))
        # print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test, y: Y_test})), "without model: {:.3f}".format(benchmark_err))
        # plt.plot(test_err)
        # plt.show()
        # print ("Total Time: %3.2fs" % float(time.time() - begin_time))

def plot_res(test_err, train_batch_loss, benchmark_err, epoch):
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    
    test_x_val = np.array(list(x * 3 for x in range(0, len(test_err))))

    plt.plot(train_batch_loss[0],train_batch_loss[1], label="Training error", c=flatui[1], alpha=0.5)
    plt.plot(test_x_val, np.array(test_err), label="Test error", c=flatui[0])
    plt.axhline(y=benchmark_err[1], linestyle='dashed', label="No-modell error", c=flatui[2])
    plt.axhline(y=0.098, linestyle='dashed', label="State of the art error", c=flatui[3])

    plt.suptitle("Model error - cold queries")
    plt.yscale('log', nonposy='clip')
    plt.xlim([0,epoch+1])
    # second_axes = plt.twinx() # create the second axes, sharing x-axis
    # second_axes.set_yticks([0.2,0.4]) # list of your y values
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(loc='upper right')
    plt.show()

def make_log_param_string(learning_rate, optimizer, batch_size, warm, layer_config):
    return "lr_%s_opt_%s_bsize_%s_warm_%s_layers_%s" % (learning_rate, optimizer, batch_size, warm, len(layer_config))

def main(job_id, params):
    warm = False
    vector_options = {'structure': True,'ged': False, 'time': False, 'sim': False,'w2v': False}
    global X_train, X_test, Y_train
    global Y_test, num_training_samples, n_input

    X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('database-iMac.log-complete', warm, vector_options)
    benchmark_err = no_modell_mean_error(Y_train, Y_test)
    Y_test = np.transpose([Y_test])
    # print benchmark_err

    # for optimizer in ['AdamOptimizer']:
    #     for learning_rate in [0.001]:
    #         for batch_size in [32]:
    layer_config = [n_input]
    for _ in range(params['layer_depth'][0]):
        layer = 'layer' + str(_)
        layer_config.append(params[layer][0])
    layer_config.append(n_classes)
    print layer_config
    # layer_config = [n_input, 48, n_classes]
    # layers = build_hidden_layers(2, 30, 50, 10)
    
    # log_param = make_log_param_string(learning_rate, optimizer, batch_size, warm, layer_config)
    # print ('Starting run for %s, optimizer: %s, batch_size: %s, warm: %s, num_layers: %s' % (log_param, optimizer, batch_size, warm, len(layer_config)))
    log_param = "Spearmint"
    optimizer = 'AdamOptimizer'
    print params
    return run_nn_model(params['learning_rate'][0], benchmark_err,log_param, optimizer, params['batch_size'][0], layer_config, params['gl'][0])
if __name__ == '__main__':
    main()
