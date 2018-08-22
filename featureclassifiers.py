#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import csv
import sys
import time
import numpy as np
import xgboost
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing
from datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler,monge_elkan, cosine, strike_a_match, soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies

def evaluate_classifier(dataset='dataset-string-similarity.txt', method='rf', training_instances=-1, polynomial=False, accuracyresults = False, results=False, permuted=True):
    num_true = 0.0
    num_false = 0.0
    num_true_predicted_true = 0.0
    num_true_predicted_false = 0.0
    num_false_predicted_true = 0.0
    num_false_predicted_false = 0.0
    timer = 0.0
    result = {}
    file = None
    if accuracyresults:
        file = open('dataset-accuracyresults-{0}.txt'.format(method),'w+')
    with open( dataset ) as csvfile:
        reader = csv.DictReader( csvfile, fieldnames=[ "s1" , "s2" , "res" , "c1" , "c2", "a1", "a2", "cc1", "cc2"], delimiter='\t' )
        for row in reader:
            if row['res'] == "TRUE": num_true += 1.0
            else: num_false += 1.0
    model1 = None
    model2 = None
    if method == 'rf':
        model1 = ensemble.RandomForestClassifier( n_estimators=600 , random_state=0 , n_jobs=2, max_depth=100)
        model2 = ensemble.RandomForestClassifier( n_estimators=600 , random_state=0 , n_jobs=2, max_depth=100)
    elif method == 'et':
        model1 = ensemble.ExtraTreesClassifier( n_estimators=600 , random_state=0 , n_jobs=2, max_depth=100)
        model2 = ensemble.ExtraTreesClassifier( n_estimators=600 , random_state=0 , n_jobs=2, max_depth=100)
    elif method == 'svm':
        model1 = svm.LinearSVC( random_state=0, C=1.0)
        model2 = svm.LinearSVC( random_state=0, C=1.0)
    elif method == 'xgboost':
        model1 = xgboost.XGBClassifier( n_estimators=3000 , seed=0 )
        model2 = xgboost.XGBClassifier( n_estimators=3000 , seed=0 )
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    print "Reading dataset..."
    with open( dataset ) as csvfile:
        reader = csv.DictReader( csvfile, fieldnames=[ "s1" , "s2" , "res" , "c1" , "c2", "a1", "a2", "cc1", "cc2"], delimiter='\t' )
        start_time = time.time()
        for row in reader:
            if row['res'] == "TRUE":
                if len(Y1) < ( ( num_true + num_false ) / 2.0 ): Y1.append(1.0)
                else: Y2.append(1.0)
            else:
                if len(Y1) < ( ( num_true + num_false ) / 2.0 ): Y1.append(0.0)
                else: Y2.append(0.0)
            row['s1'] = row['s1'].decode('utf-8')
            row['s2'] = row['s2'].decode('utf-8')
            start_time = time.time()
            sim1 = damerau_levenshtein( row['s1'] , row['s2'] )
            sim8 = jaccard( row['s1'] , row['s2'] )
            sim2 = jaro( row['s1'] , row['s2'] )
            sim3 = jaro_winkler( row['s1'] , row['s2'] )
            sim4 = jaro_winkler( row['s1'][::-1] , row['s2'][::-1] )
            sim11 = monge_elkan( row['s1'] , row['s2'] )
            sim7 = cosine( row['s1'] , row['s2'] )
            sim9 = strike_a_match( row['s1'] , row['s2'] )
            sim12 = soft_jaccard( row['s1'] , row['s2'] )
            sim5 = sorted_winkler( row['s1'] , row['s2'] )
            if permuted: sim6 = permuted_winkler( row['s1'] , row['s2'] )
            sim10 = skipgram( row['s1'] , row['s2'] )
            sim13 = davies( row['s1'] , row['s2'] )
            timer += (time.time() - start_time)
            if permuted:
                if len(X1) < ( ( num_true + num_false ) / 2.0 ): X1.append( [ sim1 , sim2 , sim3 , sim4 , sim5 , sim6 , sim7 , sim8 , sim9 , sim10 , sim11 , sim12 , sim13 ] )
                else: X2.append( [ sim1 , sim2 , sim3 , sim4 , sim5 , sim6 , sim7 , sim8 , sim9 , sim10 , sim11 , sim12 , sim13 ] )
            else:
                if len(X1) < ( ( num_true + num_false ) / 2.0 ): X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                    
    if polynomial:
        X1 = preprocessing.PolynomialFeatures().fit_transform(X1)
        X2 = preprocessing.PolynomialFeatures().fit_transform(X2)
    print "Training classifiers..."
    if training_instances > 0 :
        model1.fit( np.array(X1)[training_instances,:] , np.array(Y1)[training_instances,:] )
        model2.fit( np.array(X2)[training_instances,:] , np.array(Y2)[training_instances,:] )
    else:
        model1.fit( np.array(X1) , np.array(Y1) )
        model2.fit( np.array(X2) , np.array(Y2) )

    print "Matching records..."
    real = Y2 + Y1
    start_time = time.time()
    predicted = list( model2.predict( np.array(X1) ) ) + list( model1.predict( np.array(X2) ) )
    timer += (time.time() - start_time)
    for pos in range( len( real ) ):
        if real[pos] == 1.0:
            if predicted[pos] == 1.0:
                num_true_predicted_true += 1.0
                if accuracyresults:
                    file.write("TRUE\tTRUE\n")
            else:
                num_true_predicted_false += 1.0
                if accuracyresults:
                    file.write("TRUE\tFALSE\n")
        else:
            if predicted[pos] == 1.0:
                num_false_predicted_true += 1.0
                if accuracyresults:
                    file.write("FALSE\tTRUE\n")
            else:
                num_false_predicted_false += 1.0
                if accuracyresults:
                    file.write("FALSE\tFALSE\n")
    if accuracyresults:
        file.close()
    timer = ( timer / float( int( num_true + num_false ) ) ) * 50000.0
    acc = ( num_true_predicted_true + num_false_predicted_false ) / ( num_true + num_false )
    pre = ( num_true_predicted_true ) / ( num_true_predicted_true + num_false_predicted_true )
    rec = ( num_true_predicted_true ) / ( num_true_predicted_true + num_true_predicted_false )
    f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )
    print "Metric = Supervised Classifier :" , method.upper()
    print "Accuracy =", acc
    print "Precision =", pre
    print "Recall =", rec
    print "F1 =", f1
    print "Processing time per 50K records =", timer
    if training_instances > 0: print "Number of training instances =", training_instances
    else: print "Number of training instances =", min( len(Y1) , len(Y2) )
    print ""
    print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
    print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(method.upper(), acc, pre, rec, f1, timer)
    print ""
    print "Feature ranking"
    if method == 'rf' or method == 'et' or method == 'xgboost' :
        importances = ( model1.feature_importances_ + model2.feature_importances_ ) / 2.0
        indices = np.argsort(importances)[::-1]
        for f in range(importances.shape[0]) :
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
            if results:
                result[indices[f]] = importances[indices[f]]
    else:
        importances = ( model1.coef_.ravel() + model2.coef_.ravel() ) / 2.0
        indices = np.argsort(importances)[::-1]
        for f in range(importances.shape[0]) : print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print ""
    sys.stdout.flush()
    if results:
        return result
