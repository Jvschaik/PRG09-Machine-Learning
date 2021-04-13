"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_from_json_as_np_array(key, json_data):
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "0852893" # done: aanpassen aan je eigen studentnummer

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)

# UNSUPERVISED LEARNING

# datapunten die een x en een y hebben
kmeans_training = data.clustering_training()
# print(kmeans_training) 

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)
# print(X)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[..., 0]
y = X[..., 1]

# Done: print deze punten uit en omcirkel de mogelijke clusters
clusters = 2
kmeans = KMeans(n_clusters=clusters)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
Cx = centroids[...,0]
Cy = centroids[...,1]

plt.scatter(x, y)
plt.scatter(Cx, Cy, c='black', s=10000, alpha=0.3)

plt.axis([min(x), max(x), min(y), max(y)])
plt.show() # figure 1

# Done: ontdek de clusters mbv kmeans en teken een plot met kleurtjes
groups = kmeans.predict(X)
plt.scatter(x, y, c=groups)
plt.scatter(Cx, Cy, c='black', s=200)

plt.axis([min(x), max(x), min(y), max(y)])
plt.show()