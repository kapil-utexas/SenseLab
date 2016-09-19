import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
#B:Using scikit-learn, fit a logistic regression model to the dataset and evaluate its performance
#(accuracy) with 10-fold cross-validation. You should report performance using the accuracy measure,
#averaged across all cross-validation runs. 

data = np.loadtxt("q1_data.csv", delimiter=",")
X = data[:,(0,1)]
Y = data[:,2]
#print"DATA"
#print (data)
#print X[:10,:]

print("################################Q1 Begins###################################")
clf = LogisticRegression()
clf = clf.fit(X,Y)
scores = cross_validation.cross_val_score(clf, X, Y,scoring='accuracy', cv=10)
print("Q1 b) Logistic Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#print X
#C:Can you think of a way to improve the performance of the model while still employing the logistic
#regression algorithm? If so, describe how, include your code and present performance results. (Hint:
#think about creating new features based on F1 and F2)
#TODO : still has to increase the accuracy
pca = PCA(n_components=1)
pca.fit(X)
Xnew = pca.transform(X)
#print Xnew[:10,:]
X1 = X[:,0].reshape(-1, 1)
#print X1[:10,:]
X2 = X[:,1].reshape(-1,1)
Xnew = np.add(X1,X2) 
#print Xnew[:10,:]

clf = LogisticRegression()
scores = cross_validation.cross_val_score(clf, Xnew, Y, scoring='accuracy', cv=10)
print("Q1 C) New Features Log Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#D:Fit a Random Forest model to the dataset and based on what you find out, discuss why you think it
#performs better or worse than logistic regression
#evaluate the model by splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
clf= RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf.fit(X_train,Y_train)
#score = clf.score(X_test,Y_test)
#print score
scores = cross_validation.cross_val_score(clf, X, Y, scoring='accuracy', cv=10)
print("Q1 D) Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("################################Q1 ENDS###################################")

##########################Q1 ENDS#######################################################

#clf = LogisticRegression()
#clf.fit(X_train, Y_train)
#predicted = clf.predict(X_test)
#print predicted
#probs = clf.predict_proba(X_test)
#print probs
######################################Q2 Begins#################################################
print("################################Q2 Begins###################################")
data = np.loadtxt("q2_data.csv", delimiter=",")
X = data[:,(0,1)]
Y = data[:,2]

##Q2 b
#Using scikit-learn, fit an SVM model with a linear kernel to the dataset and evaluate its performance
#(accuracy) with 10-fold cross-validation. You should report performance using the accuracy measure,
#averaged across all cross-validation runs.
clf= SVC(kernel="linear", C=0.025)
clf.fit(X_train,Y_train)
#score = clf.score(X_test,Y_test)
#print (""score)
scores = cross_validation.cross_val_score(clf, X, Y, scoring='accuracy', cv=10)
print("Q2 B) SVM Linear Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Q2_C:Now, fit an SVM model with a non-linear kernel to the dataset and evaluate its performance
#(accuracy) with 10-fold cross-validation. You should report performance using the accuracy measure,
#averaged across all cross-validation runs
clf= SVC(gamma=2,C=1)#RBF Kernel
clf.fit(X_train,Y_train)
scores = cross_validation.cross_val_score(clf, X, Y, scoring='accuracy', cv=10)
print("Q2 C) SVM Non-Linear Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#Q2_D Fit a Random Forest model to the q2_data.csv dataset and discuss why you think it performs better
#or worse than SVM (with linear and non-linear kernel)
#TODO:  how the number of trees in the forest affect the performance results.

clf= RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf.fit(X_train,Y_train)
scores = cross_validation.cross_val_score(clf, X, Y, scoring='accuracy', cv=10)
print("Q2 D) Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
######################################Q2 Ends#################################################

# This code is copied from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py
# And modified a little bit to print train and test scores mean 
# This code defines the function for plotting learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    
    train_sizes = np.linspace
        default is np.linspace(.1, 1.0, 5)
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve( estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    print("train_scores_mean:")
    print(np.mean(train_scores, axis=1))
    print("test_scores_mean:")
    print (test_scores_mean)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#############################################MY CODE for Q3###################################################
print("################################Q3 Begins###################################")

Xy = np.loadtxt("q3_data.csv", delimiter=",")
X = Xy[:,:20]
y = Xy[:,20]
#train_sizes_abs, train_scores, test_scores = learning_curve(LinearSVC(C=10),X, y, cv=5, n_jobs=1, train_sizes = np.linspace(.05, 0.2, 5))
#print(np.mean(train_scores, axis=1))
#test_scores_mean = np.mean(test_scores, axis=1)
#print(test_scores_mean = np.mean(test_scores, axis=1))
#print("test_scores_mean:")
#print (test_scores_mean)

title = "Given : Learning Curve for Linear SVC"
plot_learning_curve(LinearSVC(C=10),title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=1,train_sizes = np.linspace(.05, 0.2, 5))

#Provide source code showing one way to reduce overfitting in this example without modifying the
#classification algorithm (i.e., LinearSVC). (Hint: Think about training data size, regularization and SVM
#parameterization).
# np.linspace(Fraction of total data used as the plot step(a),
#             Fraction of total data used as training(b),
#             Total number of points(c))
a = .05  
b = .4  # Training Data Size 
c =  5  
title = "Reduced overfitting: Learning Curve for Linear SVC"
plot_learning_curve(LinearSVC(C=10),title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=1,train_sizes = np.linspace(a,b,c))
plt.show()


