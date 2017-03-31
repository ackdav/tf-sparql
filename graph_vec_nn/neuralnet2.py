import tensorflow as tf
tf.reset_default_graph()

from tensorflow.python.client import timeline
import re, ast, os, sys, time
from random import sample
import numpy as np

from nn_helper import *

LOGDIR = 'logs/neuralnet2/'

# Parameters
training_epochs = 300

n_classes = 1
n_input = 62
X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
no_model_error = 0.
# tf Graph input
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None,1], name="y")

def multilayer_perceptron(x, layer_config, name="neuralnet"):
    '''
    sets up neural network in a dynamic way to test effectiveness of height and depth of network
    code from: https://pythonprogramming.net/community/262/TensorFlow%20For%20loop%20to%20set%20weights%20and%20biases/
    '''
    layers = {}
    layers_compute = {}

    with tf.name_scope(name):
        for i in range(1, len(layer_config)):
            new_layer = {'weights': tf.Variable(tf.random_normal([layer_config[i-1], layer_config[i]], 0, 0.1)),
                        'biases': tf.Variable(tf.random_normal([layer_config[i]], 0, 0.1))}
            layers[i-1] = new_layer

            with tf.name_scope("weights"):
                tf.summary.histogram("w_l"+str(i)+"_summary", new_layer['weights'])

            with tf.name_scope("biases"):
                tf.summary.histogram("b_l"+str(i)+"_summary", new_layer['biases'])

            l = tf.add(tf.matmul(x if i == 1 else layers_compute[i-2], layers[i-1]['weights']), layers[i-1]['biases'])
            
            with tf.name_scope(name):
                l = tf.nn.relu(l) if i != len(layer_config)-1 else l
                # l = tf.nn.dropout(l, keep_rate) if i != len(layer_config)-1 else l

            layers_compute[i-1] = l

    lastlayer = len(layers_compute)-1
    return layers_compute[lastlayer]

def run_nn_model(learning_rate, log_param, optimizer, batch_size, layer_config):
    begin_time = time.time()
    prediction = multilayer_perceptron(x, layer_config)
    
    #profiling options to generate a trace
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    with tf.name_scope("RMSE"):
        cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))), name="RMSE")
        tf.summary.scalar("RMSE", cost)
    with tf.name_scope("relativeMeanError"):
        perc_err = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y, prediction)), tf.reduce_mean(y)))
        tf.summary.scalar("relativeMeanError", perc_err)

    with tf.name_scope("train"):
        if optimizer == 'AdagradOptimizer':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(perc_err)
        if optimizer == 'FtrlOptimizer':
            optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(perc_err)
        if optimizer == 'AdadeltaOptimizer':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(perc_err)
        if optimizer == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(perc_err)

    # merge all summaries into a single "operation" which we can execute in a session 
    summary_op = tf.summary.merge_all()
    
    # Launch the graph
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOGDIR + log_param , graph=tf.get_default_graph())

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
                _, c, p, s = sess.run([optimizer, cost, prediction, summary_op], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                # <-- PROFILING --> , options=run_options, run_metadata=run_metadata
                # tl = timeline.Timeline(run_metadata.step_stats)
                # ctf = tl.generate_chrome_trace_format()
                # with open('timeline.json', 'w+') as f:
                #     f.write(ctf)
                avg_cost += c / total_batch

            # sample prediction
            label_value = batch_y
            estimate = p
            err = label_value-estimate
            # Display logs per epoch step
            if epoch % 3 == 0:
                # sess.run(assignment, feed_dict={x: X_test, y: Y_test})
                writer.add_summary(s, epoch)

                print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
                print ("[*]----------------------------")
                for i in xrange(3):
                    print ("label value:", label_value[i], \
                        "estimated value:", estimate[i])
                print ("[*]============================")
                sys.stdout.flush()

            if epoch % 100 == 0:
                # mean_relative_error = tf.divide(tf.to_float(tf.reduce_sum(perc_err)), Y_test.shape[0])
                print ("RMSE: {:.3f}".format(cost.eval({x: X_test, y: Y_test})))
                print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test, y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(Y_train, Y_test)))
                sess.run([optimizer, summary_op], feed_dict={x: X_test, y: Y_test})
                saver.save(sess, LOGDIR + os.path.join(log_param, "model.ckpt"), i)

        # <-- profiling -->
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)
        print ("RMSE: {:.3f}".format(cost.eval({x: X_test, y: Y_test})))
        print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test, y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(Y_train, Y_test)))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))

def make_log_param_string(learning_rate, optimizer, batch_size, warm, layer_config):
    return "lr_%s_opt_%s_bsize_%s_warm_%s_layers_%s" % (learning_rate, optimizer, batch_size, warm, len(layer_config))

# def build_hidden_layers(num_total_layers, min_size, end_size, step=30):
#     configs = []
#     for i in xrange(1, num_total_layers+1, 1):
#         hidden = [min_size] * i
#         for idx, j in enumerate(hidden):
#             for k in xrange(j, end_size+1, step):
#                 print k, idx, hidden
#                 hidden[idx] = k
#                 test = hidden
#                 configs = configs + [hidden]
#     return configs

def main():
    warm = True
    global X_train, X_test, Y_train, Y_test, num_training_samples, n_input
    X_train, X_test, Y_train, Y_test, num_training_samples, n_input = load_data('random200k.log-result', warm, 'hybrid')
    
    #setup to find optimal nn
    for optimizer in ['AdadeltaOptimizer']:
        for learning_rate in [0.01]:
            for batch_size in [16]:

                layer_config = [n_input, 100, 150, 200, n_classes]
                # layers = build_hidden_layers(2, 30, 50, 10)
                
                log_param = make_log_param_string(learning_rate, optimizer, batch_size, warm, layer_config)
                print ('Starting run for %s, optimizer: %s, batch_size: %s, warm: %s, num_layers: %s' % (log_param, optimizer, batch_size, warm, len(layer_config)))

                run_nn_model(learning_rate, log_param, optimizer, batch_size, layer_config)
if __name__ == '__main__':
    main()
