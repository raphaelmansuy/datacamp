# Unsupervised Learning

## Unsupervised learning

* Unsupervised learning = finds patterns in data
  * E.g. **_clustering_** customers by their purchase
  * Compressing the data using purchase patterns (**_dimension reduction_**)

## Unsupervised learning vs Supervised learning

* Supervised learning = learning with labelled data
* Discovers patterns / classification without labeled data

## k-means clustering

* Finds clusters of samples
* Number of clusters must be specified
* Implemented in **sklearn** (scikit-learn)

### k-means in action

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(samples)

KMeans(algorithm='auto',...)

labels = model.predict(samples)

print(labels)
```

### Scatter plot

```python
import matplotlib.pyplot as plt

xs = samples[:,0]
ys = samples[:,2]

ply.scatter(xs,yx,c=labels)
```

### Exercice 1 : clustering data

```python
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)
```

### Exercice 2: display clustering

```python
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()

```

## Evaluating a clustering

### Cross tabulation with pandas

#### Align labes and species

```python
import pandas as pd

df = pd.DataFrame({'labels':labels,'species':species})

print(df)

ct = pd.crosstab(df['labels'],df['species'])

print(ct)
```

### Inertia measures the quality of clustering

* Measures how spread out the clusters are (_lower_ is better)
* Distance from each sample to centroid of its cluster

```python

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(samples)

print(model.inertia_)

```

### How many clusters to choose?

* A good clustering has tight clusters (so low intertia)
* ... but not to many clusters
* Choose an "elbow" in the inertia plot
* Were inertia begins to to decrease more slowly

#### Exercice 3 : how many clusers clusters of grain

```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
```

```python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples,varieties)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)

```

## Tranforming features for better clusterings

### Sample clustering the wines

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)

# Cluster vs varieties

df = pd.DataFrame('labels':labels,
    'varieties':varieties)

ct = pd.crosstabl(df['labels'],df['varieties'])

print(ct)
```

### Feature variances

* The wine featues have very different variances
* Variance of feature mesures spread of it's value 

### StandardScaler

* In kmeans: feature variance = feature influence
* **StandardScaler** transforms each feature to have mean 0 and variance 1
* Features are said to be "standardized"

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=True,with_mean=True,with_std=True)

scaler.fit(samples)

samples_scaled = scaler.transform(samples)
```
### Using a pipeline to combine scaling and clustering

```python

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(samples)

labels = pipeline.predict(samples)

df = pd.DataFrame({'labels':labels,'varieties':varieties})

ct = pd.crosstab(df['labels'],df['varieties'])

print(ct)
```

### sklearn preprocessing steps examples

* StandardScaler
* MaxAbsScaler
* Normalizer

### Exercice pipeline with scaling

```python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels,'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['species'])

# Display ct
print(ct)

```

### Exerice : Clustering stocks using KMeans

```python
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)
# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))

```

## Visualizing hierarchies

### Poweful visualization to communicate insight

* "t-SNE": create a 2D map of a DataSet
* "Hierachical clustering"

#### Hierarchical clustering
 
For example Eurovision vote by country (Dendogram)

* Every country begins in a separate cluster
* At each step, the two closest clusters are merged
* Continue until all countries in a single cluster
* This is "agglomerative" hierarchical clustering

```python
import matplotlib.pyplot as plt
from scipy.cluser.hierarchy import linkage,dendrogram

mergins = linkage(samples,method='complete')

dendogram(mergins,labels=country_names,
  leaf_rotation=90,
  leaf_font_size=5)

plt.show()
```

#### Exercice

```python
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples,method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

```

```python
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements,method='complete')

# Plot the dendrogram
dendogram(mergings,labels=companies,leaf_rotation=90,leaf_font_size=6)
plt.show()
```

#### Cluster labels in hierarchical clustering

* Extracting cluster labels
  * Use **fcluster** method


```python
from scipy.cluster.hierarchy import linkage

mergins = linkage(samples,method='complete')

from scipy.cluster.hierarchy import fcluster

labels = fcluster(mergins,15,criterion='distance')

print(labels)
```

* Aligning cluster labels with country naes

```python
import pandas as pd
pairs = pd.DataFrae({
  'labels': labels,
  'countries': country_names
})

print(pairs.sort_values('labels'))
```

#### Exercice single linkage (closest point between 2 clusters )

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples,method='single')

# Plot the dendrogram
dendrogram(mergings,labels=country_names,leaf_rotation=90,leaf_font_size=6)
plt.show()

```

#### Exercice : extracting the cluster labels

```python
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings,6,criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)

```

### t-SNE for 2 dimensional maps

* t-SNE = "t-distributed stochatisc neighbors embedding"
* Maps samples to 2D space (or 3D) 
* Map approximately preserves nearness of samples
* Great for inspecting datasets

#### t-SNE in sklearn

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = TSNE(learning_rate=100)

transformed = model.fit_transform(samples)

xs = transformed[:,0]
ys = transformed[:,1]

plt.scatter(xs,ys,c=species)
plt.show()
```

#### Exercice : t-SNE visualization of grain dataset

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()
```

#### Exercice t-SNE of the stock market data

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

plt.show()

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
```

## Visualize PCA (Principal Component Analysis)