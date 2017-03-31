import sys, re, ast
from random import sample
import numpy as np

def load_data(log_file, warm, feature_mode, train_size_ratio=0.8):
    query_data = []

    with open(log_file) as f:
        for line in f:
            query_line = line.strip('\n')
            query_line = query_line.split('\t')
            query_vec = unicode(query_line[1])
            query_vec = ast.literal_eval(query_vec)
            #TEMP-fix - TODO: adjust with new dataset
            query_vec = query_vec[0:-1]
            
            if (warm):
                query_vec.insert(len(query_vec),query_line[2])
            if not (warm):
                query_vec.insert(len(query_vec),query_line[3])
            query_data.append(query_vec)

    n_input = len(query_data[0])-1
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
    load_data('random200k.log-result', True, 'hybrid')

if __name__ == '__main__':
    main()