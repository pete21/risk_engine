import os
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

# load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# train model
clf = svm.LinearSVC(max_iter=1000)
clf.fit(X, y)

# persistent model
path = os.path.join(os.path.dirname(__file__), 'iris_model.pickle')
joblib.dump(clf, path)

# test code to check the saved model
clf = joblib.load(path)
assert clf.predict([[5.0, 3.6, 1.3, 0.25]])[0] == 0
