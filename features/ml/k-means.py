from sklearn import datasets, metrics
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

mnist = datasets.load_digits()
images = mnist.images
data_size = len(images)
# Preprocessing images
images = images.reshape(len(images), -1)
labels = mnist.target
# Initialize Logistic Regression
clustering = KMeans(n_clusters=10, init='k-means++', n_init=10)
# Training the data on only 75% of the dataset. Rest of the 25% will be used in testing the KMeans Clustering
clustering.fit(images[:int((data_size / 4) * 3)])
# Print the centers of the different clusters
print(clustering.labels_)
# Testing the data
predictions = clustering.predict(images[int((data_size / 4)):])
#plt.show(predictions)