from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']


print(iris.data)

gmm = GaussianMixture(n_components=4, random_state=0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

print(gmm_cluster_labels)

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting
print(X)
print(1e-4)