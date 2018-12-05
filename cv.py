#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@author: yyl
#@time: 2017/12/23 13:27
import sys
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
import math
import scipy.io as sio


final_fuse = "concat"

conv_1_shape = '4*4*32'
pool_1_shape = 'None'

conv_2_shape = '4*4*64'
pool_2_shape = 'None'

conv_3_shape = '4*4*128'
pool_3_shape = 'None'

conv_4_shape = '1*1*13'
pool_4_shape = 'None'

window_size = 128
n_lstm_layers = 2

# lstm full connected parameter
n_hidden_state = 32
print("\nsize of hidden state", n_hidden_state)
n_fc_out = 1024
n_fc_in = 1024

dropout_prob = 0.5
np.random.seed(32)

norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True

cnn_suffix        =".mat_win_128_cnn_dataset.pkl"
rnn_suffix        =".mat_win_128_rnn_dataset.pkl"
label_suffix    =".mat_win_128_labels.pkl"

data_file    =sys.argv[1]
arousal_or_valence = sys.argv[2]
with_or_without = sys.argv[3]

dataset_dir = "/home/yyl/ijcnn/deap_shuffled_data/"+with_or_without+"_"+arousal_or_valence+"/"
###load training set
with open(dataset_dir + data_file + cnn_suffix, "rb") as fp:
    cnn_datasets = pickle.load(fp)
with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
    rnn_datasets = pickle.load(fp)
with open(dataset_dir + data_file + label_suffix, "rb") as fp:
    labels = pickle.load(fp)
    labels = np.transpose(labels)
    print("loaded shape:",labels.shape)
lables_backup = labels
print("cnn_dataset shape before reshape:", np.shape(cnn_datasets))
cnn_datasets = cnn_datasets.reshape(len(cnn_datasets), window_size, 9,9, 1)
print("cnn_dataset shape after reshape:", np.shape(cnn_datasets))
one_hot_labels = np.array(list(pd.get_dummies(labels)))

labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

# shuffle data
index = np.array(range(0, len(labels)))
np.random.shuffle(index)

cnn_datasets   = cnn_datasets[index]
rnn_datasets   = rnn_datasets[index]
labels  = labels[index]

print("**********(" + time.asctime(time.localtime(time.time())) + ") Load and Split dataset End **********\n")
print("**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions Begin: **********\n")

# input parameter
n_input_ele = 32
n_time_step = window_size

input_channel_num = 1
input_height = 9
input_width = 9

n_labels = 2
# training parameter
lambda_loss_amount = 0.5
training_epochs = 70

batch_size = 200


# kernel parameter
kernel_height_1st = 4
kernel_width_1st = 4

kernel_height_2nd = 4
kernel_width_2nd = 4

kernel_height_3rd = 4
kernel_width_3rd = 4

kernel_height_4th = 1
kernel_width_4th = 1

kernel_stride = 1
conv_channel_num = 32

# algorithn parameter
learning_rate = 1e-4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.elu(tf.layers.batch_normalization(conv2d(x, weight, kernel_stride)))

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions End **********")
print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")

# input placeholder
cnn_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='cnn_in')
rnn_in = tf.placeholder(tf.float32, shape=[None, n_time_step, n_input_ele], name='rnn_in')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

###########################################################################################
# add cnn parallel to network
###########################################################################################
# first CNN layer
with tf.name_scope("conv_1"):
    conv_1 = apply_conv2d(cnn_in, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
    print("conv_1 shape:", conv_1.shape)
# second CNN layer
with tf.name_scope("conv_2"):
    conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num * 2,kernel_stride)
    print("conv_2 shape:", conv_2.shape)
# third CNN layer
with tf.name_scope("conv_3"):
    conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4,kernel_stride)
    print("conv_3 shape:", conv_3.shape)
# depth concatenate
with tf.name_scope("depth_concatenate"):
    cube = tf.reshape(conv_3,[-1,9,9,conv_channel_num * 4 * window_size])
    print("cube shape:", cube.shape)
# fourth CNN layer
with tf.name_scope("conv_4"):
    conv_4 = apply_conv2d(cube, kernel_height_4th, kernel_width_4th, conv_channel_num * 4 * window_size, 13,kernel_stride)
    print("\nconv_4 shape:", conv_4.shape)

# flatten (13*9*9) cube into a 1053 vector.
shape = conv_4.get_shape().as_list()
conv_3_flat = tf.reshape(conv_4, [-1, shape[1] * shape[2] * shape[3]])

cnn_out_fuse = conv_3_flat
###########################################################################################
# add lstm parallel to network
###########################################################################################
# rnn_in         ==>    [batch_size, n_time_step, n_electrode]
shape = rnn_in.get_shape().as_list()
# rnn_in_flat     ==>    [batch_size*n_time_step, n_electrode]
rnn_in_flat = tf.reshape(rnn_in, [-1, shape[2]])
# fc_in     ==>    [batch_size*n_time_step, n_electrode]
rnn_fc_in = apply_fully_connect(rnn_in_flat, shape[2], n_fc_in)
# lstm_in    ==>    [batch_size, n_time_step, n_fc_in]
lstm_in = tf.reshape(rnn_fc_in, [-1, n_time_step, n_fc_in])
# define lstm cell
cells = []
for _ in range(n_lstm_layers):
    with tf.name_scope("LSTM_"+str(n_lstm_layers)):
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in,dtype=tf.float32, time_major=False)

output = tf.unstack(tf.transpose(output, [1, 0, 2]), name='lstm_out')
rnn_output = output[-1]
###########################################################################################
# fully connected
###########################################################################################
# rnn_output ==> [batch, fc_size]
shape_rnn_out = rnn_output.get_shape().as_list()
# fc_out ==> [batch_size, n_fc_out]
lstm_fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)
# keep_prob = tf.placeholder(tf.float32)
lstm_fc_drop = tf.nn.dropout(lstm_fc_out, keep_prob)
###########################################################################################
# fuse parallel cnn and lstm
###########################################################################################
print("final fuse method: concat")
fuse_cnn_rnn = tf.concat([cnn_out_fuse, lstm_fc_drop], axis=1)

fuse_cnn_rnn_shape = fuse_cnn_rnn.get_shape().as_list()
print("\nfuse_cnn_rnn:", fuse_cnn_rnn_shape)
# readout layer
y_ = apply_readout(fuse_cnn_rnn, fuse_cnn_rnn_shape[1], n_labels)
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
    tf.summary.scalar('cost_with_L2',cost)
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')
    tf.summary.scalar('cost',cost)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy',accuracy)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

merged = tf.summary.merge_all()
logdir = "my_tensorboard"
train_writer = tf.summary.FileWriter("log/"+logdir+"/train")
test_writer = tf.summary.FileWriter("log/"+logdir+"/test")

fold = 5
for curr_fold in range(fold):
    fold_size = cnn_datasets.shape[0]//fold
    indexes_list = [i for i in range(len(cnn_datasets))]
    indexes = np.array(indexes_list)
    split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
    split = np.array(split_list)
    cnn_test_x = cnn_datasets[split] 
    test_y = labels[split]
    rnn_test_x = rnn_datasets[split]

    split = np.array(list(set(indexes_list)^set(split_list)))
    cnn_train_x = cnn_datasets[split]
    rnn_train_x = rnn_datasets[split]
    train_y = labels[split]
    train_sample = train_y.shape[0]
    print("training examples:", train_sample)
    test_sample = test_y.shape[0]
    print("test examples    :",test_sample)
    # set train batch number per epoch
    batch_num_per_epoch = math.floor(cnn_train_x.shape[0]/batch_size)+ 1

    # set test batch number per epoch
    accuracy_batch_size = batch_size
    train_accuracy_batch_num = batch_num_per_epoch
    test_accuracy_batch_num = math.floor(cnn_test_x.shape[0]/batch_size)+ 1

    # print label
    one_hot_labels = np.array(list(pd.get_dummies(lables_backup)))
    print(one_hot_labels)

    with tf.Session(config=config) as session:
        train_writer.add_graph(session.graph)
        count_cost = 0
        train_count_accuracy = 0
        test_count_accuracy = 0

        session.run(tf.global_variables_initializer())
        train_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_loss_save = np.zeros(shape=[0], dtype=float)
        train_loss_save = np.zeros(shape=[0], dtype=float)
        for epoch in range(training_epochs):
            print("learning rate: ",learning_rate)
            cost_history = np.zeros(shape=[0], dtype=float)
            for b in range(batch_num_per_epoch):
                start = b* batch_size
                if (b+1)*batch_size>train_y.shape[0]:
                    offset = train_y.shape[0] % batch_size
                else:
                    offset = batch_size
                # print("start:",start,"end:",start+offset)
                cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
                cnn_batch = cnn_batch.reshape(len(cnn_batch) * window_size, 9, 9, 1)
                rnn_batch = rnn_train_x[start:(start + offset), :, :]
                batch_y = train_y[start:(start + offset), :]
                _ , c = session.run([optimizer, cost],
                                   feed_dict={cnn_in: cnn_batch, rnn_in: rnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                              phase_train: True})
                cost_history = np.append(cost_history, c)
                count_cost += 1
            if (epoch % 1 == 0):
                train_accuracy = np.zeros(shape=[0], dtype=float)
                test_accuracy = np.zeros(shape=[0], dtype=float)
                test_loss = np.zeros(shape=[0], dtype=float)
                train_loss = np.zeros(shape=[0], dtype=float)

                for i in range(train_accuracy_batch_num):
                    start = i* batch_size
                    if (i+1)*batch_size>train_y.shape[0]:
                        offset = train_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    train_cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
                    train_cnn_batch = train_cnn_batch.reshape(len(train_cnn_batch) * window_size, 9, 9, 1)

                    train_rnn_batch = rnn_train_x[start:(start + offset), :, :]
                    train_batch_y = train_y[start:(start + offset), :]

                    tf_summary,train_a, train_c = session.run([merged,accuracy, cost],
                                                   feed_dict={cnn_in: train_cnn_batch, rnn_in: train_rnn_batch,
                                                              Y: train_batch_y, keep_prob: 1.0, phase_train: False})
                    train_writer.add_summary(tf_summary,train_count_accuracy)
                    train_loss = np.append(train_loss, train_c)
                    train_accuracy = np.append(train_accuracy, train_a)
                    train_count_accuracy += 1
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                      np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
                train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
                train_loss_save = np.append(train_loss_save, np.mean(train_loss))

                if(np.mean(train_accuracy)<0.8):
                    learning_rate=1e-4
                elif(0.8<np.mean(train_accuracy)<0.85):
                    learning_rate=5e-5
                elif(0.85<np.mean(train_accuracy)):
                    learning_rate=5e-6

                for j in range(test_accuracy_batch_num):
                    start = j * batch_size
                    # print(start)
                    if (j+1)*batch_size>test_y.shape[0]:
                        offset = test_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    test_cnn_batch = cnn_test_x[start:(start + offset), :, :, :, :]
                    test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, 9, 9, 1)

                    test_rnn_batch = rnn_test_x[start:(start + offset), :, :]
                    test_batch_y = test_y[start:(start + offset), :]

                    tf_test_summary,test_a, test_c = session.run([merged,accuracy, cost],
                                                 feed_dict={cnn_in: test_cnn_batch, rnn_in: test_rnn_batch, Y: test_batch_y,
                                                            keep_prob: 1.0, phase_train: False})
                    test_writer.add_summary(tf_test_summary,test_count_accuracy)
                    test_accuracy = np.append(test_accuracy, test_a)
                    test_loss = np.append(test_loss, test_c)
                    test_count_accuracy += 1 
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                      np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
                test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
                test_loss_save = np.append(test_loss_save, np.mean(test_loss))
            # reshuffle
            index = np.array(range(0, len(train_y)))
            np.random.shuffle(index)
            cnn_train_x=cnn_train_x[index]
            rnn_train_x=rnn_train_x[index]
            train_y=train_y[index]

            # learning_rate decay
            if(np.mean(train_accuracy)<0.9):
                learning_rate=1e-4
            elif(0.9<np.mean(train_accuracy)<0.95):
                learning_rate=5e-5
            elif(0.99<np.mean(train_accuracy)):
                learning_rate=5e-6

        test_accuracy = np.zeros(shape=[0], dtype=float)
        test_loss = np.zeros(shape=[0], dtype=float)
        test_pred = np.zeros(shape=[0], dtype=float)
        test_true = np.zeros(shape=[0, 2], dtype=float)
        test_posi = np.zeros(shape=[0, 2], dtype=float)
        for k in range(test_accuracy_batch_num):
            start = k * batch_size
            if (k+1)*batch_size>test_y.shape[0]:
                offset = test_y.shape[0] % batch_size
            else:
                offset = batch_size
            test_cnn_batch = cnn_test_x[start:(start + offset), :, :, :, :]
            test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, 9, 9, 1)
            test_rnn_batch = rnn_test_x[start:(start + offset), :, :]
            test_batch_y = test_y[start:(start + offset), :]

            test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi],
                                                         feed_dict={cnn_in: test_cnn_batch, rnn_in: test_rnn_batch,
                                                                    Y: test_batch_y, keep_prob: 1.0, phase_train: False})
            test_t = test_batch_y

            test_accuracy = np.append(test_accuracy, test_a)
            test_loss = np.append(test_loss, test_c)
            test_pred = np.append(test_pred, test_p)
            test_true = np.vstack([test_true, test_t])
            test_posi = np.vstack([test_posi, test_r])
        test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
        test_true_list = tf.argmax(test_true, 1).eval()
        # recall
        test_recall = recall_score(test_true, test_pred_1_hot, average=None)
        # precision
        test_precision = precision_score(test_true, test_pred_1_hot, average=None)
        # f1 score
        test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
        # confusion matrix
        # confusion_matrix = confusion_matrix(test_true_list, test_pred)
        print("********************recall:", test_recall)
        print("*****************precision:", test_precision)
        print("******************f1_score:", test_f1)
        # print("**********confusion_matrix:\n", confusion_matrix)

        print("(" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
              "Final Test Accuracy: ", np.mean(test_accuracy))

        result = pd.DataFrame(
            {'epoch': range(1, epoch + 2), "train_accuracy": train_accuracy_save, "test_accuracy": test_accuracy_save,
             "train_loss": train_loss_save, "test_loss": test_loss_save})

        ins = pd.DataFrame({'conv_1': conv_1_shape,'conv_2': conv_2_shape,'conv_3': conv_3_shape, 'conv_4': conv_4_shape,
                            'final_fuse': final_fuse,'rnn fc in': n_fc_in, 'rnn fc out': n_fc_out,
                            'hidden_size': n_hidden_state, 'accuracy': np.mean(test_accuracy),
                            'keep_prob': 1 - dropout_prob,'sliding_window': window_size, "epoch": epoch + 1, "norm": norm_type,
                            "learning_rate": learning_rate, "regularization": regularization_method,
                            "train_sample": train_sample, "test_sample": test_sample,"batch_size":batch_size}, index=[0])
        summary = pd.DataFrame({'recall': test_recall, 'precision': test_precision,'f1_score': test_f1})
        writer = pd.ExcelWriter(
            "./results/cv_"+arousal_or_valence+"/"+ data_file +"_"+str(curr_fold)+".xlsx")
        ins.to_excel(writer, 'condition', index=False)
        result.to_excel(writer, 'result', index=False)
        summary.to_excel(writer, 'summary', index=False)
        # fpr, tpr, auc
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        i = 0
        for key in one_hot_labels:
            fpr[key], tpr[key], _ = roc_curve(test_true[:, i], test_posi[:, i])
            roc_auc[key] = auc(fpr[key], tpr[key])
            roc = pd.DataFrame({"fpr": fpr[key], "tpr": tpr[key], "roc_auc": roc_auc[key]})
            roc.to_excel(writer, str(key), index=False)
            i += 1
        writer.save()
        # save model
        model_dict= {}
        parameter_count=0
        
        for variable in tf.trainable_variables():
            print(variable.name,"-->",variable.get_shape())
            count = 1
            for dim in variable.get_shape().as_list():
                count = count * dim
            parameter_count = parameter_count+count
            model_dict[variable.name]=session.run(variable)
        sio.savemat("PCRNN_model_"+str(parameter_count)+".mat",model_dict)
        print("----------------------------------------------------------------")
        print("------------------total parameters",parameter_count,"-----------------------")
        print("----------------------------------------------------------------")
        
        # save model
        '''
        saver = tf.train.Saver()
        saver.save(session,
                   "./result/cnn_rnn_parallel/tune_rnn_layer/" + output_dir + "/model_" + output_file)
        '''
        print("**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN End **********\n")
train_writer.close()
