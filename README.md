# Classification Testing

**Classify truth-tellers versus liars based on their cluster distributions using several different classification methods from scikit-learn. Designed for use with the ROC-HCI deception dataset, to contrast the temporal modeling of Hidden Markov Models.**

*Author: Matt Levin*


#### Usage

`python3 clf.py [-m Method] [-i InputFolder] [-k NumberFolds] [-d NumberClusters] [-s RandomSeed]`


#### Valid Method Options
* **SVM** - uses sklearn.svm.SVC as classifier          *(Support Vector Classifier)*
* **MLP** - uses sklearn.neural_network.MLPClassifier   *(Multilayer Perceptron Classifier)*
* **GNB** - uses sklearn.naive_bayes.GaussianNB         *(Gaussian Naive Bayes Classifier)*
* **BNB** - uses sklearn.naive_bayes.BernoulliNB        *(Bernoulli Naive Bayes Classifier)*
* **DT**  - uses sklearn.tree.DecisionTreeClassifier    *(Decision Tree Classifier)*
* **KNN** - uses sklearn.neighbors.KNeighborsClassifier *(K-Nearest Neighbors Classifier)*


#### Example Usage
* `python3 clf.py -m MLP -i test -k 10 -s 9999` 
  * Uses MLPClassifier on test data folder with 10 folds and random state 9999.
* `python3 clf.py -h`
  * Shows usage/help message.
