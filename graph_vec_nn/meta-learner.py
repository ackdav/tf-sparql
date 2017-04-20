import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

DIMS = 10  # Dimensions of the parabola
LAYERS = 2
STATE_SIZE = 20
TRAINING_STEPS = 20  # This is 100 in the paper

scale = tf.random_uniform([DIMS], 0.5, 1.5)
# This represents the network/function we are trying to optimize,
# the `optimizee' as it's called in the paper.
# Actually, it's more accurate to think of this as the error
# landscape.
def f(x):
    # x = scale*x
    # return tf.reduce_sum(x*x)
    return tf.square(x)

# def g_sgd(gradients, state, learning_rate=0.1):
#     return -learning_rate*gradients, state

# def g_rms(gradients, state, learning_rate=0.1, decay_rate=0.99):
#     if state is None:
#         state = tf.zeros(DIMS)
#     state = decay_rate*state + (1-decay_rate)*tf.pow(gradients, 2)
#     update = -learning_rate*gradients / (tf.sqrt(state)+1e-5)
#     return update, state

cell = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in range(LAYERS)])
cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
cell = tf.make_template('cell', cell)

def g_rnn(gradients, state):
    # Make a `batch' of single gradients to create a 
    # "coordinate-wise" RNN as the paper describes.
    print type(gradients)
    gradients = tf.expand_dims(gradients, axis=1)
 
    if state is None:
        state = [[tf.zeros([DIMS, STATE_SIZE])] * 2] * LAYERS
    # print "gradients"
    # print gradients
    # print "state"
    # print state
    update, state = cell(gradients, state)
    # Squeeze to make it a single batch again.
    return tf.squeeze(update, axis=[1]), state

def optimize(loss):
    optimizer = tf.train.AdamOptimizer(0.0001)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.)
    return optimizer.apply_gradients(zip(gradients, v))

initial_pos = tf.random_uniform([DIMS], -1., 1.)
def learn(optimizer):
    losses = []
    x = initial_pos
    state = None
    for _ in range(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        print loss
        grads, = tf.gradients(loss, x)

        update, state = optimizer(grads, state)
        x += update
    return losses

# sgd_losses = learn(g_sgd)
# rms_losses = learn(g_rms)

rnn_losses = learn(g_rnn)
sum_losses = tf.reduce_sum(rnn_losses)
apply_update = optimize(sum_losses)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

ave = 0
for i in range(4000):
    err, _ = sess.run([sum_losses, apply_update])
    ave += err
    if i % 1000 == 0:
        print(ave / 1000 if i!=0 else ave)
        ave = 0
print(ave / 1000)
x = np.arange(TRAINING_STEPS)

for _ in range(3): 
    rnn_l = sess.run([rnn_losses])
    # p1, = plt.plot(x, sgd_l, label='SGD')
    p3, = plt.plot(x, rnn_l, label='RNN')
    # p2, = plt.plot(x, rms_l, label='RMS')
    # plt.legend(handles=[p1,p3, p2])
    plt.title('Losses')
    plt.show()