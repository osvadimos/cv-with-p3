import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition
digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
fig, plot = plt.subplots()
plot.scatter(X_pca[:, 0], X_pca[:, 1])
plot.set_xticks(())
plot.set_yticks(())
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
plt.show()