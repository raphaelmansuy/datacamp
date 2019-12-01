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

### Dimension reduction
 
 * More efficient storage and computation
 * Remove less-informative "noise" features
 * ... wich cause problems for predictions tasks, e.g. classification, regression

### Principal Component Analysis

* PCA = "Principal Component Analysis"
* Fundamental dimension reduction technique
* First step "decorrelation" 
  * PCA aligns data with axes
  * Shifs data samples so they have mean 0
    * No information is lost
* Second step reduces dimension

```python
from sklearn.decomposition import PCA

model = PCA()

model.fit(samples)

transformed = model.transform(samples)

print(transformed)
```

#### Exercice correlated data in nature

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width,length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width,length)

# Display the correlation
print(correlation)

```

#### Exercise decorrelating the grain measurements with PCA

```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```

### Intrinsic dimension

* Intrinsic dimension = number of features needed to approximate the dataset
* Essential idea behind dimension reduction
* What is the most compact representation of the samples ?
* Can be detected with PCA

PCA identifies intrinsic dimension

* PCA identifies intrinsic dimension when samples have _any number_ of features
* Intrinsic dimension = number of PCA features with significant variance

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(samples)

features = range(pca.n_components_)

plt.bar(features,pca.explained_variance_)
plt.xticks(features)A

plt.ylabel('variance')
plt.xlabel('PCA feature')

plt.show()

```

### Exercice : the first principal component

```python
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
```

#### Exercice : variance of the PCA feature

```python
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

```

### Dimension reduction with PCA

* Represeents same data, using less features
* Important part of machine-learning pipelines
* Can be done with PCA

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(samples)

transformed = pca.transform(samples)

import matplotlib.pyplot as plt

xs = transformed[:,0]
ys = transformed[:,1]

plt.scatter(xs,ys,c=species)
plt.show()
```

Dimension reduction with PCA

* Discards low variance PCA features
* Assumes the high variance features are informative
* Assumption typlicalle holds in practice (e.g. for iris)


#### Word frequency arrays

  * Rows represent documents, columns represents words
  * Entries measure presence of each word in each document
  * ... measure using "tf-idf"

#### Sparse arrays and csr_matrix

  * Array is "sparse" : most entries are zero
  * Can use **scipy.sparse.csr_matrix** instead of NumPy array
  * skit-learn PCA doesn't support csr_matrix
  * Use scikit-learn **TruncatedSVD** instead

```python
from sklearn.decomposition import TruncatedSVD 

model = TruncatedSVD(n_components=1)

model.fit(documents)

transformed = model.transform(documents)
```

#### Dimension reduction of the fish measurements

```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

```

#### Exercice: A tf-idf word-frequency array

```python

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)
```

```python
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)
``# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))

```

## Non-negative matrix factorization (NMF)

* NMF = "non-negative matrix factorization"
* Dimension reduction technique
* NMF models are _interpretable_ (unlike PCA)
* All samples features must be non-negative (>= 0)

### Example with word-frequency array

```python
from sklearn.decomposition import NMF

model = NMF(n_components=2)

model.fit(samples)

nmf_features = model.transform(samples) # NMF has components just like PCA has principal components

# Dimension of components = dimension of samples
# Entries are non-negative
# NMF entries are non-negatives

```

 NMF fits to non-negative data, only

 * Word frequencies in each documents
 * Images encoded as array
 * Audio spectrograms
 * Purchase histories on e-commerce sites

#### NMF applied to Wikipedia articles

```python
# Import NMF
from sklearn.decomposition import NMF 

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features,index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

```

### NMF learns interpretable parts

Example:

* Word-frequency array articles (tf-idf)
* 20,000 scientific articles (rows)
* 800 words (columns)

Applying NMF to the articles:

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=10)

print(nmf.components_.shape)
```

NMF components:

* For documents:
  * NMF components represent topics
  * NMF features combine topics into documents 
* For images:
  * NMF component are parts of images

Visualize 
```python

# Reshape vector 1D representation of an image into a 2D matrix

bitmap = sample.reshape(2,3)

from matplotlib import pyplot as plt
plt.imshow(bitmap,cmap='gray',interpolation='nearest')
plt.show()
```

#### Exercice : NMF learns topics of documents
```python
# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_,columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3,:]

# Print result of nlargest
print(component.nlargest())
```

#### Exercice : Explore the LED digits dataset

```python
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
```

### Exercie NMF learn part of the image

```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)
```

#### Exercice PCA doesn't learn part

```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
```
## Building recommender systerm with NMF

#### Exercice calculate similarities

```python
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo',:]

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
```

#### Exercice recommend musical artist

```python
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen',:]

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())

```


