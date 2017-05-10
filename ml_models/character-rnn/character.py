import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
import reader


data = open('input.txt', 'r').read()
chars = list(set(data)) #convert input to char list
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique' % (data_size,vocab_size)

char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

#convert array of chars to array of vocab indices
def c2i(inp):
    return map(lambda c:char_to_ix[c], inp)

def i2c(inp):
    return map(lambda c:ix_to_char[c], inp)

hidden_size = 100 #size of hidden layer of neurons
seq_length = 25 #number of steps to unroll RNN for / maximum context the RNN is expected to retain while training
learning_rate = 0.001
batch_size = 50
num_epochs = 100
checkpoint_file = "rnn-cell-model.ckpt"

x = tf.placeholder(tf.int32, shape=([batch_size, seq_length]), name="x")
y = tf.placeholder(tf.int32, shape=([batch_size, seq_length]), name="y")
seed = tf.placeholder(tf.int32, [1], name='seed')
rnn_input = tf.one_hot(seed, vocab_size)
# x_oh & y_oh are of shape [batch_size, seq_length, vocab_size] now.
x_oh = tf.one_hot(indices=x, depth=vocab_size)
y_oh = tf.one_hot(indices=y, depth=vocab_size)

#gen x & y => y is just x shifted one to the right
def gen_epoch_data(num_epochs, batch_size, seq_length):
    for i in range(num_epochs):
        yield reader.ptb_iterator(data, batch_size, seq_length)

data = c2i(data)

def train_network(graph, num_epochs, batch_size, seq_length, checkpoint):
    tf.set_random_seed(2345)
    prev_epoch_loss = 1e50
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.isfile(checkpoint):
            g['saver'].restore(sess, checkpoint)

        for idx, epoch in enumerate(gen_epoch_data(num_epochs, batch_size, seq_length)):
            # training_state = np.zeros([batch_size, hidden_size], dtype=float)
            training_state = None
            epoch_loss = 0.0
            batches = 0
            for batchIdx, (x, y) in enumerate(epoch):
                batches = batches + 1
                feed_dict = {
                    graph['x'] : x,
                    graph['y'] : y,
                }
                if training_state is not None:
                    feed_dict[graph['init_state']] = training_state

                _, _, rnn_inputs, rnn_outputs, init_state, training_state, \
                total_loss, train_step = sess.run(
                    [
                        graph['x_oh'],
                        graph['y_oh'],
                        graph['rnn_inputs'],
                        graph['rnn_outputs'],
                        graph['init_state'],
                        graph['final_state'],
                        graph['total_loss'],
                        graph['train_step'],
                    ],
                    feed_dict
                )

                if (batchIdx % 10 == 0):
                    inp_seed = np.array([x[0][0]])

                    print '\n'
                    print '--- SAMPLE BEGIN ---'
                    num_chars = 100
                    ixes = []
                    # sstate = np.zeros([hidden_size, 1])

                    if training_state is not None:
                        feed_dict[graph['init_state']] = training_state
                    for j in range(num_chars):
                        prob_r, sstate = sess.run([graph['by'], graph['init_state']], 
                            feed_dict={seed:inp_seed, graph['x']:x, graph['y']:y })
                        # prob_r, sstate = sess.run([prob, sample_state], feed_dict={seed:inp_seed, init_state:sstate, x:x_i})

                        prob_r = map(abs, prob_r)
                        prob_r = np.asarray([prob_r])
                        ix = np.random.choice(vocab_size, p=prob_r.ravel())
                        ixes.append(ix)
                        inp_seed = np.array([ix])

                    print ''.join(i2c(ixes))
                    print '--- SAMPLE END ---'

                epoch_loss += total_loss
                '''
                if batchIdx % 5 == 0:
                    print 'Epoch:', idx, 'Batch:', batchIdx, 'Loss:', total_loss
                    print init_state
                    print '---'
                    print training_state
                '''
            epoch_loss /= batches
            print 'Epoch:', idx, 'Average epoch loss:', epoch_loss
            if epoch_loss < prev_epoch_loss:
                g['saver'].save(sess, checkpoint_file)
            prev_epoch_loss = epoch_loss
            losses.append(epoch_loss)
    return losses

def build_graph(batch_size, seq_length, vocab_size, state_size, learning_rate):

    # Basically converts the input into [seq_length, batch_size, vocab_size].
    rnn_inputs = tf.unstack(x_oh, axis=1)
    # y_oh is also of the same shape as x_oh
    # i.e., [seq_length, batch_size, vocab_size].
    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, seq_length, 1)]

    # print tf.shape(x_oh), tf.shape(x)

    # Creates a hidden state of size state_size per batch.
    cell = tf.contrib.rnn.BasicRNNCell(state_size)
    # init_state would be a vector of shape [batch_size, state_size].
    init_state = cell.zero_state(batch_size, tf.float32)

    # The RNN Cell abstracts away this calculation:
    # Hi = tanh(X Wxh + Hi-1 Whh + bh)
    # Wxh is of shape [vocab_size, hidden_size].
    # Whh is of shape [hidden_size, hidden_size].
    # The shape of the rnn_output would be [seq_length, batch_size, hidden_size].
    # The rnn() method will basically iterate over all the batches.
    # [[batch_size, vocab_size], [batch_size, vocab_size], ...].
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        Why = tf.get_variable('Why', [state_size, vocab_size])
        by = tf.get_variable('by', [vocab_size], initializer=tf.constant_initializer(0.0))


    logits = [tf.matmul(rnn_output, Why) + by for rnn_output in rnn_outputs]
    loss_weights = [tf.ones([batch_size]) for i in range(seq_length)]
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        x_oh = x_oh,
        y_oh = y_oh,
        by = logits,
        rnn_inputs = rnn_inputs,
        rnn_outputs = rnn_outputs,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        saver = tf.train.Saver()
    )

g = build_graph(batch_size, seq_length, vocab_size, hidden_size, learning_rate)
losses = train_network(g, num_epochs, batch_size, seq_length, checkpoint_file)
# f = open('rnn-cell-losses.txt', 'w')
# for loss in losses:
#     f.write(str(loss) + '\n')
# f.close()
plt.plot(losses)
plt.show()