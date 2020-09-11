# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
# Change the Linear Model to the Passive Agressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier

classifier = PassiveAggressiveClassifier(random_state=0)
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Avoid getting warning for matplotlib colors
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# Visualising the Training set results
from matplotlib.colors import ListedColormap



x_set, y_set = x_train, y_train
first, second = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
    np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    first,
    second,
    classifier.predict(np.array([first.ravel(), second.ravel()]).T).reshape(
        first.shape
    ),
    alpha=0.75,
    cmap=ListedColormap(("pink", "lightgreen")),
)
plt.xlim(first.min(), first.max())
plt.ylim(second.min(), second.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        x_set[y_set == j, 0],
        x_set[y_set == j, 1],
        c=ListedColormap(("red", "green"))(i),
        label=j,
    )
plt.title("Classifier (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap

x_set, y_set = x_test, y_test
first, second = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
    np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    first,
    second,
    classifier.predict(np.array([first.ravel(), second.ravel()]).T).reshape(
        first.shape
    ),
    alpha=0.75,
    cmap=ListedColormap(("pink", "lightgreen")),
)
plt.xlim(first.min(), first.max())
plt.ylim(second.min(), second.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        x_set[y_set == j, 0],
        x_set[y_set == j, 1],
        c=ListedColormap(("red", "green"))(i),
        label=j,
    )
plt.title("Classifier (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
