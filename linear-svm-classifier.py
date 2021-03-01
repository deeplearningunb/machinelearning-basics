# Classification template

# Importing the libraries
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC # C-SVM based on libsvm
classifier = SVC(C=1.0, kernel='linear', degree=1, random_state=0)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Obtain Metrics
Tp, Fp, Fn, Tn = cm.reshape(1, -1, order='F').squeeze()
# https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
accuracy = (Tp+Tn) / (Tp+Fp+Tn+Fn)
precision = (Tp) / (Tp+Fp)
recall = (Tp) / (Tp+Fn)
f1_score = 2 * (precision*recall) / (precision+recall)
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrics = [accuracy, precision, recall, f1_score]
session_results = pd.DataFrame(metrics, index=metrics_names)
print(session_results)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
lX1 = np.linspace(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, num = 650)
lX2 = np.linspace(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, num = 650)
X1, X2 = np.meshgrid(lX1, lX2)
X3 = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
X_it = sc.inverse_transform(np.array([X1.ravel(), X2.ravel()]).T)
X1, X2 = X_it[:,0].reshape(X1.shape), X_it[:,1].reshape(X1.shape),
plt.contourf(X1, X2, X3, alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
X_set_it = sc.inverse_transform(X_set)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set_it[y_set == j, 0], X_set_it[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolors='k')
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig(join('figures', 'linear_svm_train.png'), dpi=300, bbox_inches='tight')
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
lX1 = np.linspace(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, num = 650)
lX2 = np.linspace(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, num = 650)
X1, X2 = np.meshgrid(lX1, lX2)
X3 = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
X_it = sc.inverse_transform(np.array([X1.ravel(), X2.ravel()]).T)
X1, X2 = X_it[:,0].reshape(X1.shape), X_it[:,1].reshape(X1.shape),
plt.contourf(X1, X2, X3, alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
X_set_it = sc.inverse_transform(X_set)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set_it[y_set == j, 0], X_set_it[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, edgecolors='k')
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig(join('figures', 'linear_svm_test.png'), dpi=300, bbox_inches='tight')
plt.show()
