import tensorflow as tf
tf.reset_default_graph()

from tensorflow.python.client import timeline
import re, ast, os, sys, time
from random import sample
import numpy as np

LOGDIR = 'logs/neuralnet2/'

# Parameters
training_epochs = 100

n_classes = 1
n_input = 63
X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
y_vals = np.array([])
no_model_error = 0.
# tf Graph input
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None,1], name="y")

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

def load_data(log_file, warm, feature_mode):
    query_data = []
    global Y_test
    global X_test
    global Y_train
    global X_train
    global y_vals
    global num_training_samples
    global n_input
    with open(log_file) as f:
        for line in f:
            query_line = line.strip('\n')
            query_line = query_line.split('\t')
            query_vec = unicode(query_line[1])
            query_vec = ast.literal_eval(query_vec)

            if (warm):
                query_vec.insert(len(query_vec),query_line[2])
            if not (warm):
                query_vec.insert(len(query_vec),query_line[3])

            query_data.append(query_vec)

    y_vals = np.array([ float(x[n_input]) for x in query_data])

    for l_ in query_data:
        del l_[-1]
        if feature_mode == 'structural':
            l_=l_[0:51]
        elif feature_mode == 'ged':
            l_=l_[51:]
        else:
            l_=l_
        n_input = len(l_)

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

    Y_test = np.transpose([Y_test])

def no_modell_mean_error(y_vals):
    mean_ = y_vals.mean()
    mean_sum = 0.
    for y_ in y_vals:
        mean_error = abs(y_- mean_) / mean_
        mean_sum += mean_error
    return mean_sum / y_vals.shape[0]

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

            # with tf.name_scope("weights"):
            #     tf.summary.histogram("w_l"+str(i)+"_summary", new_layer['weights'])

            # with tf.name_scope("biases"):
            #     tf.summary.histogram("b_l"+str(i)+"_summary", new_layer['biases'])

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
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
        if optimizer == 'FtrlOptimizer':
            optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(cost)
        if optimizer == 'AdadeltaOptimizer':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
        if optimizer == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
                c, p, s = sess.run([cost, prediction, summary_op], feed_dict={x: batch_x, y: batch_y})
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
                print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test, y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(y_vals)))
                sess.run([optimizer, summary_op], feed_dict={x: X_test, y: Y_test})
                saver.save(sess, LOGDIR + os.path.join(log_param, "model.ckpt"), i)

        # <-- profiling -->
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)
        print ("RMSE: {:.3f}".format(cost.eval({x: X_test, y: Y_test})))
        print ("relative error with model: {:.3f}".format(perc_err.eval({x: X_test, y: Y_test})), "without model: {:.3f}".format(no_modell_mean_error(y_vals)))
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
    load_data('random200k.log-result', warm, 'hybrid')
    
    #setup to find optimal nn
    for optimizer in ['AdadeltaOptimizer']:
        for learning_rate in [0.01]:
            for batch_size in [100]:

                layer_config = [n_input, 10, 10, 5, n_classes]
                # layers = build_hidden_layers(2, 30, 50, 10)
                
                log_param = make_log_param_string(learning_rate, optimizer, batch_size, warm, layer_config)
                print ('Starting run for %s, optimizer: %s, batch_size: %s, warm: %s, num_layers: %s' % (log_param, optimizer, batch_size, warm, len(layer_config)))

                run_nn_model(learning_rate, log_param, optimizer, batch_size, layer_config)
if __name__ == '__main__':
    main()
