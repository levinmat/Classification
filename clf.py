#!/usr/local/bin/python3

"""
Classify truth-tellers vs bluffers based on their cluster distributions using several different
classification methods. Designed for use with the ROC-HCI deception dataset.

Author: Matt Levin

Usage: 
   python3 clf.py [-m Method] [-i InputFolder] [-k NumberFolds] [-d NumberClusters] [-s RandomSeed]

Valid Method Options:
    SVM - uses sklearn.svm.SVC as the classifier      (Support Vector Machine Classifier)
    MLP - uses sklearn.neural_network.MLPClassifier   (Multilayer Perceptron Classifier)
    GNB - uses sklearn.naive_bayes.GaussianNB         (Gaussian Naive Bayes Classifier)
    BNB - uses sklearn.naive_bayes.BernoulliNB        (Bernoulli Naive Bayes Classifier)
    DT  - uses sklearn.tree.DecisionTreeClassifier    (Decision Tree Classifier)
    KNN - uses sklearn.neighbors.KNeighborsClassifier (K-Nearest Neighbors Classifier)
    
Ex:
   python3 clf.py -m MLP -i test -k 10 -s 9999
   
"""

import argparse     # Parse arguments
import glob         # Browse file folders
import csv          # Write results to CSV
import numpy as np  # Vector operations
import time         # Record time of execution

# Cross validation wrapper
from sklearn.model_selection import cross_val_score

# Classifiers
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


""" Loads datafiles from the input folder and returns X and y """
def load_data(inputfolder, num_clusters):
    # X = Cluster Distribution 
    # y = 0/1 for Truther/Bluffer 
    X, y = [], []
    
    if 'input/' not in inputfolder:
        inputfolder = 'input/' + inputfolder
    
    # Truthers (y = 0)
    for filename in glob.glob(inputfolder + '/truthers/*'):
        with open(filename) as f:
            points = np.array([int(x) for x in f.read().split()], dtype=int)
            counts = np.bincount(points)
            counts.resize(num_clusters)
            X.append(counts / float(len(points)))
            y.append(0)
            f.close()
      
    # Bluffers (y = 1)          
    for filename in glob.glob(inputfolder + '/bluffers/*'):
        with open(filename) as f:
            points = np.array([int(x) for x in f.read().split()], dtype=int)
            counts = np.bincount(points)
            counts.resize(num_clusters)            
            X.append(counts / float(len(points)))
            y.append(1)
            f.close()
    
    return np.array(X), np.array(y)
   
                             
""" Returns a classifier object for the given method """
def create_classifier(method, seed):
    if method == 'svm':
        return SVC(random_state=seed)
    if method == 'mlp':
        # Could use some fine-tuning on the layer sizes - lbfgs/relu seems to work well
        return MLPClassifier(hidden_layer_sizes=(50,100,75), 
                            solver='lbfgs', activation="relu",
                            max_iter=2000, tol=1e-4, alpha=0.0001, 
                            shuffle=True, random_state=seed)
    if method == 'gnb':
        return GaussianNB()
    if method == 'bnb':
        return BernoulliNB()
    if method == 'dt':
        return DecisionTreeClassifier(random_state=seed)
    if method == 'knn':
        return KNeighborsClassifier()

    raise RuntimeError('Invalid method argument\n\tOptions = (SVM|GNB|BNB|MLP|DT|KNN)' +
                       '\n\tUse -h or --help flags to show proper usage.')


""" Generates easily seperable sample data with given number of outliers """
def gen_test_data(num_outliers=0):
    for i in range(100):
        t = open('input/test/truthers/{}.txt'.format(i), 'w+')
        b = open('input/test/bluffers/{}.txt'.format(i), 'w+')
        # Outliers get the opposite values, should be classified incorrectly
        # Technically not 'outliers' but go against the rest of the data        
        if i in range(num_outliers):
            for _ in range(200):
                t.write('0\n1\n2\n')                          
                b.write('2\n3\n4\n')
        else:
            for _ in range(200):
                b.write('0\n1\n2\n')
                t.write('2\n3\n4\n')
        t.close()
        b.close()


""" Main Method - Parse args, perform cross-validation, record results """
if __name__ == '__main__':
    #gen_test_data(num_outliers=10) # Generates sample data for testing
    
    help = """
    Classify truth-tellers vs bluffers by their cluster distributions using various classification methods.
    
    Usage: 
       python3 clf.py [-m Method] [-i InputFolder] [-k NumberFolds] [-d NumberClusters] [-s RandomSeed]
    
    Valid Method Options:
       SVM - uses sklearn.svm.SVC as the classifier      (Support Vector Machine Classifier)
       MLP - uses sklearn.neural_network.MLPClassifier   (Multilayer Perceptron Classifier)
       GNB - uses sklearn.naive_bayes.GaussianNB         (Gaussian Naive Bayes Classifier)
       BNB - uses sklearn.naive_bayes.BernoulliNB        (Bernoulli Naive Bayes Classifier)
       DT  - uses sklearn.tree.DecisionTreeClassifier    (Decision Tree Classifier)
       KNN - uses sklearn.neighbors.KNeighborsClassifier (K-Nearest Neighbors Classifier)
    
    Ex:
       python3 clf.py -m SVM -i test -k 10 -s 9999
    
    """
    parser = argparse.ArgumentParser(description=help,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', metavar='Method', default='svm', type=str,
                        help='The method of classification used, options=(SVM|GNB|BNB|MLP)')
    parser.add_argument('-i', metavar='InputFolder', type=str, default='test', 
                        help='Input Folder (must contain truthers/bluffers subfolders)')
    parser.add_argument('-k', metavar='NumberFolds', type=int, default=5,
                        help='Number of folds to use in cross-validation')
    parser.add_argument('-s', metavar='RandomSeed', help='Random Seed', type=int, default=101)
    parser.add_argument('-d', metavar='NumberClusters', type=int, default=5,
                        help='Number of clusters used in cluster sequences')
    args = parser.parse_args()
    
    print('Method:      ' + args.m)
    print('InputFolder: ' + args.i)
    print('NumFolds:    ' + str(args.k))
    print('NumClusters: ' + str(args.d))
    print('RandomSeed:  ' + str(args.s))
    
    print('\nBeginning cross-validation...\n')
    
    # Loads the data from the infolder (X = cluster distribution | y = 0/1 for T/B)
    X, y = load_data(args.i, args.d)
    # Generates the classifier object based on the given method
    clf = create_classifier(args.m.lower(), args.s)
    # Performs crossvalidation and gets the average classification accuracy score
    score = np.average(cross_val_score(clf, X, y, cv=args.k)) * 100
    
    
    print('Average Accuracy = {}%'.format(score))
    
    
    # Save Result as: [Time, Method, InputFolder, NumFolds, RandomSeed, Score]
    f = open('results.csv', 'a+')
    writer = csv.writer(f)
    writer.writerow([time.ctime(), args.m.lower(), args.i, args.k, args.s, score])   
    f.close()
    
    print('\nProgram Complete.')
    
    