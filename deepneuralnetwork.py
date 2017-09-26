#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import csv
import time
import unicodedata
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Merge, Dense, Dropout, GRU, Bidirectional
from keras.layers.core import Masking

def deep_neural_net_gru(train_data_1, train_data_2, train_labels, test_data_1, test_data_2, test_labels, max_len,
                        len_chars, hidden_units=60, bidirectional=True, n = 1):
    early_stop = EarlyStopping(monitor='loss', patience=0, verbose=1)
    checkpointer = ModelCheckpoint(filepath="checkpoint" + str(n) +".hdf5", verbose=1, save_best_only=True)
    gru1 = GRU(hidden_units, consume_less='gpu', return_sequences=True)
    gru2 = GRU(hidden_units, consume_less='gpu', return_sequences=False)
    if bidirectional:
        gru1 = Bidirectional(gru1)
        gru2 = Bidirectional(gru2)
    left_branch = Sequential(name="left_branch")
    left_branch.add(Masking(mask_value=0, input_shape=(max_len, len_chars), name="mask1"))
    left_branch.add(gru1)
    left_branch.add(Dropout(0.01))
    left_branch.add(gru2)                                       #
    left_branch.add(Dropout(0.01))                              # single bidirectional
    right_branch = Sequential(name="right_branch")
    right_branch.add(Masking(mask_value=0, input_shape=(max_len, len_chars), name="mask2"))
    right_branch.add(gru1)
    right_branch.add(Dropout(0.01))
    right_branch.add(gru2)                                      # single bidirectional
    right_branch.add(Dropout(0.01))                             #
    con_layer = Sequential(name="con_layer")
    con_layer.add(Merge([left_branch, right_branch], mode='concat', name="merge_con"))
    mul_layer = Sequential(name="mul_layer")
    mul_layer.add(Merge([left_branch, right_branch], mode='mul', name="merge_mul"))
    dif_layer = Sequential(name="dif_layer")
    dif_layer.add(Merge([left_branch, right_branch],
                        mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0], name="merge_dif"))
    final_model = Sequential(name="final_model")
    final_model.add(Merge([con_layer, mul_layer, dif_layer], mode='concat',name="merge_threeconcat")) # only concat
    #final_model.add(con_layer)
    final_model.add(Dropout(0.01))
    final_model.add(Dense(hidden_units, activation='relu'))
    final_model.add(Dropout(0.01))
    #final_model.add(Dense(hidden_units, activation='relu'))     #
    #final_model.add(Dropout(0.01))                              # single rlu
    #final_model.add(Dense(hidden_units, activation='relu'))     #
    #final_model.add(Dropout(0.01))                              #
    final_model.add(Dense(1, activation='sigmoid'))
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit([train_data_1, train_data_2], train_labels, validation_data=([test_data_1, test_data_2], test_labels),
                    callbacks=[early_stop,checkpointer], nb_epoch=20)
    start_time = time.time()
    aux = final_model.predict_classes([test_data_1, test_data_2]).ravel()
    return aux, (time.time() - start_time)


def evaluate_deep_neural_net(dataset='dataset-string-similarity.txt', method='gru', training_instances=-1,
                             bidirectional=True, hiddenunits=60):
    max_seq_len = 40
    num_true = 0.0
    num_false = 0.0
    num_true_predicted_true = 0.0
    num_true_predicted_false = 0.0
    num_false_predicted_true = 0.0
    num_false_predicted_false = 0.0
    timer = 0.0
    with open(dataset) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2"], delimiter='\t')
        for row in reader:
            if row['res'] == "TRUE":
                num_true += 1.0
            else:
                num_false += 1.0
    XA1 = []
    XB1 = []
    XC1 = []
    Y1 = []
    XA2 = []
    XB2 = []
    XC2 = []
    Y2 = []
    start_time = time.time()
    print "Reading dataset... " + str(start_time - start_time)
    with open(dataset) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2"], delimiter='\t')
        start_time = time.time()
        for row in reader:
            if row['res'] == "TRUE":
                if len(Y1) < ((num_true + num_false) / 2.0):
                    Y1.append(1)
                else:
                    Y2.append(1)
            else:
                if len(Y1) < ((num_true + num_false) / 2.0):
                    Y1.append(0)
                else:
                    Y2.append(0)
            row['s1'] = row['s1'].decode('utf-8')
            row['s2'] = row['s2'].decode('utf-8')
            row['s1'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s1'] + u'|')), encoding='utf-8')
            row['s2'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s2'] + u'|')), encoding='utf-8')
            if len(XA1) < ((num_true + num_false) / 2.0):
                XA1.append(row['s1'])
                XB1.append(row['s2'])
            else:
                XA2.append(row['s1'])
                XB2.append(row['s2'])
        print "Dataset read... " + str(time.time() - start_time)
    Y1 = np.array(Y1, dtype=np.bool)
    Y2 = np.array(Y2, dtype=np.bool)
    chars = list(set(list([val for sublist in XA1 + XB1 + XA2 + XB2 for val in sublist])))
    char_labels = {ch: i for i, ch in enumerate(chars)}
    aux1 = np.memmap("temporary-file-dnn-1-" + method, mode="w+", shape=(len(XA1), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XA1):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XA1 = aux1
    aux1 = np.memmap("temporary-file-dnn-2-" + method, mode="w+", shape=(len(XB1), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XB1):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XB1 = aux1
    aux1 = np.memmap("temporary-file-dnn-3-" + method, mode="w+", shape=(len(XA2), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XA2):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XA2 = aux1
    aux1 = np.memmap("temporary-file-dnn-4-" + method, mode="w+", shape=(len(XB2), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XB2):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XB2 = aux1
    print "Temporary files created... " + str(time.time() - start_time)
    print "Training classifiers..."

    if training_instances <= 0: training_instances = min(len(Y1), len(Y2))

    aux1, time1 = deep_neural_net_gru(train_data_1=XA1[0:training_instances, :, :],
                                      train_data_2=XB1[0:training_instances, :, :],
                                      train_labels=Y1[0:training_instances, ], test_data_1=XA2, test_data_2=XB2,
                                      test_labels=Y2, max_len=max_seq_len, len_chars=len(chars),
                                      bidirectional=bidirectional, hidden_units=hiddenunits, n=1)
    aux2, time2 = deep_neural_net_gru(train_data_1=XA2[0:training_instances, :, :],
                                      train_data_2=XB2[0:training_instances, :, :],
                                      train_labels=Y2[0:training_instances, ], test_data_1=XA1, test_data_2=XB1,
                                      test_labels=Y1, max_len=max_seq_len, len_chars=len(chars),
                                      bidirectional=bidirectional, hidden_units=hiddenunits, n=2)
    timer += time1 + time2
    print "Matching records..."
    real = list(Y1) + list(Y2)
    file = open("dataset-dnn-accuracy","w+")
    predicted = list(aux2) + list(aux1)
    for pos in range(len(real)):
        if float(real[pos]) == 1.0:
            if float(predicted[pos]) == 1.0:
                num_true_predicted_true += 1.0
                file.write("TRUE\tTRUE\n")
            else:
                num_true_predicted_false += 1.0
                file.write("TRUE\tFALSE\n")
        else:
            if float(predicted[pos]) == 1.0:
                num_false_predicted_true += 1.0
                file.write("FALSE\tTRUE\n")
            else:
                num_false_predicted_false += 1.0
                file.write("FALSE\tFALSE\n")

    timer = (timer / float(int(num_true + num_false))) * 50000.0
    acc = (num_true_predicted_true + num_false_predicted_false) / (num_true + num_false)
    pre = (num_true_predicted_true) / (num_true_predicted_true + num_false_predicted_true)
    rec = (num_true_predicted_true) / (num_true_predicted_true + num_true_predicted_false)
    f1 = 2.0 * ((pre * rec) / (pre + rec))
    file.close()
    print "Metric = Deep Neural Net Classifier :", method.upper()
    print "Bidirectional :", bidirectional
    print "Accuracy =", acc
    print "Precision =", pre
    print "Recall =", rec
    print "F1 =", f1
    print "Processing time per 50K records =", timer
    print "Number of training instances =", training_instances
    print ""
    os.remove("temporary-file-dnn-1-" + method)
    os.remove("temporary-file-dnn-2-" + method)
    os.remove("temporary-file-dnn-3-" + method)
    os.remove("temporary-file-dnn-4-" + method)
    sys.stdout.flush()