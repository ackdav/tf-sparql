import sys, re, ast, itertools
from random import sample
import numpy as np

def adjust_rnn_test_arrays(X_test, Y_test, sequence_length, input_dimension):
    enlarged_batch = []
    for id, obj in enumerate(X_test):
        if id > 2:
            objn = []
            objn.insert(0, obj.tolist())
            try:
                objn.insert(0,X_test[id-1].tolist())
                # print len(X_test[id-1].tolist())
                objn.insert(0,X_test[id-2].tolist())
                objn.insert(0,X_test[id-3].tolist())
                # obj = np.append(obj, batch_x[id
                enlarged_batch.append( list(itertools.chain.from_iterable(objn))) 

            except:
                print "wrong length" + str(id)
    X_test = np.asarray(enlarged_batch)
    X_test = X_test.reshape([X_test.shape[0], sequence_length, input_dimension])
    # Y_test = Y_test[3:]
    return (X_test, Y_test)

def load_data(log_file, warm, vector_options, train_size_ratio=0.8):
    query_vectors = []
    with open(log_file) as f:
        for line in f:
            query_v = []
            query_line = line.strip('\n')
            query_line = query_line.split('\t')

            if vector_options['structure']:
                query_structure = ast.literal_eval(unicode(query_line[0]))
                query_v += query_structure
            # if vector_options['time']:
            #     query_structure = ast.literal_eval(unicode(query_line[1]))
            #     query_v += query_structure
            if vector_options['ged']:
                query_ged = ast.literal_eval(unicode(query_line[1]))
                query_v += query_ged
            if vector_options['sim']:
                query_sim = ast.literal_eval(unicode(query_line[2]))
                query_v += query_sim
            if vector_options['w2v']:
                query_w2v = ast.literal_eval(unicode(query_line[3]))
                query_v += query_w2v
            time_warm = query_line[4]
            time_cold = query_line[5]

            # query_vec = unicode(query_line[1])
            # query_vec = ast.literal_eval(query_vec)
            #TEMP-fix - TODO: adjust with new dataset
            # query_vec = query_vec[0:-1]
            if (warm):
                query_v.insert(len(query_v), time_warm)
            if not (warm):
                query_v.insert(len(query_v), time_cold)
            query_vectors.append(query_v)

    y_vals = np.array([ float(x[len(query_vectors[0])-1]) for x in query_vectors])

    for l_ in query_vectors:
        del l_[-1]
        n_input = len(l_)

    x_vals = np.array(query_vectors)

    # split into test and train
    l = len(x_vals)
    f = int(round(l*train_size_ratio))
    indices = sample(range(l), f)

    X_train = x_vals[indices].astype('float32')
    X_test = np.delete(x_vals, indices, 0).astype('float32')
    Y_train = y_vals[indices].astype('float32')
    Y_test = np.delete(y_vals, indices, 0).astype('float32')

    num_training_samples = X_train.shape[0]
    X_train = np.nan_to_num(normalize_cols(X_train))
    X_test = np.nan_to_num(normalize_cols(X_test))

    # Y_train = np.transpose([Y_train])
    Y_test = np.transpose([Y_test])
    return (X_train, X_test, Y_train, Y_test, num_training_samples, n_input)

def no_modell_mean_error(Y_test, Y_train):
    Y_test = np.transpose([Y_test])
    y_vals = np.vstack((Y_test, Y_train))
    mean_ = y_vals.mean()
    mean_sum = 0.
    for y_ in y_vals:
        mean_error = abs(y_- mean_) / mean_
        mean_sum += mean_error
    return float(mean_sum.sum() / y_vals.shape[0])

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

def main():
    vector_options = {'structure': False, 'time': True, 'ged': False,'sim': False,'w2v': False}
    load_data('database.log-complete', True, vector_options)

if __name__ == '__main__':
    main()