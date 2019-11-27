# Supervised learning with scikit-learn

## What is a classification problem

A classification problem is a problem in witch for a set of data point, we assign a label defined as a category to each point.

## What is a supervised learning process ?

Supervised learning is a learning process in which an algorithm is trained to create a function that predicts a variable based on a set feature and an observed variable. The resulting function can then be used to predict a target value for a set of features. 

## What is an unsupervised learning process ?

An unsupervised learning process is a process for a set of unlabeled data, an algorithm try to classify each point as belonging to one category.

### classification

In a classification problem the target value to predict is discrete value.

### regression 

In a regression problem the target value to predict is a continuous value

### Using scikit-learn to fit a classifier

```Python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target'])
```

```Python
iris['data].shape
```

```Shell
(150,4)
```

```Python
iris['target'].shape
````

```Shell
(150,)
```

### Predicting on unlabeled data

```Python
prediction = knn.predict(X_new)
```


#### Exercice 1

```Python
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

```
#### Exercice 2
```Python
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party',axis=1)

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
````

### Measuring model performance

In classification, accuracy is a commonly used metric.

* Accuracy = Fraction of correct predictions
* How to calculate accuracy
  * We need to split the data into training set and test set
  * Fit/train the data on the training set
  * Make prediction on the test set
  * Compare prediction with the knowns labels

```Python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)

knn = KNeighborsClaissifier(n_neighbors=8)
knn.fit(X_Train,y_train)

y_pred = knn.predict(X_test)
print(\"Test set predictions:\\n {}",format(y_pred))
knn.score(X_test,y_test)

```

### Impact of model complexity

* Larger k = smoother decision boundary = less complex model
* Smaler k = more complex model = can lead to overfeating

#### Exercice 4
```Python 
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```

### Exercice 5 : test a model and calculate the acuracy using test split

```Python
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
``` 

#### Exercice 6 : influence of the value of k for the acuracy of the model

```Python
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

### Regression

In a regression problem the target value to predict is a continuous value.

```Python
# importing the libraries
import pandas as pd
# reading the dataset
boston = pd.read_csv('boston.csv')
print(boston.head())
# Create the feature dataset and the target datset
# features dataset
X = boston.drop('MEDV,axis=1).values
# target dataset
y = boston['MEDV].values
```

## Predicting house value from a single feature

```Python
X_rooms = X[:5] # Get the number of rooms columns

y = y.reshape(-1,1)
X_rooms = X.rooms.reshape(-1,1)

plt.scatter(X_rooms,y)
plt.ylabel('Value of house / 1000 $')
plt.xlabel('Number of rooms')
plt.show()
```

### Fitting a regression model

```Python
import numpy as np
from sklearn import linear_model
reg = linear_model.linearRegression()
reg.fit(X_rooms,y)

prediction_space = np.linspace(min(X_Rooms),max(X_Rooms)).reshape(-1,1)
plt.scatter(X_Rooms,y,color='blue')
plt.plot(prediction_space,reg.predict(prediction_space),color='black',linewidth=3)
plt.show()
```

#### Exercice 6, preparing the data for linear regression

```Python
# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
```

#### Display correlation heatmap with seaborns

```Python
import matplotlib.pyplot as plt
import seaborns as sns

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
```

### Regression mechanics

* y = ax +b
  * y = target
  * x = single feature
  * a,b = parameters of model

How to chose a and b ?

  * Define an error functions for any given line
    * Choose the line that minimizes the error function (loss function)

* Linear regression in higher dimensions
  * y = a1x1+a2x2+b    
  * y = a1x1+a2x2+a3x3+ ... + an+xn + b

#### Linear regressions on all features

```Python
from sklearn.model_selection from train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,radom_state=42)

reg_all = linear_model.linearRegression()
reg_all.fit(X_train,y_train)

y_pred = reg_all.predict(X_test)
reg_all.score(X_test,y_test)

```

#### Exercice 7

```Python
# Import LinearRegression
from sklearn import linear_model

# Create the regressor: reg
reg = linear_model.LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
```


#### Exercice 8
```Python
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
```

### Cross validation

* Model performance is dependant on way the data is split
* Not representative of the model's ability to generalize
* Solution: Cross-validation ! k-fold cross validation (the dataset is split in k folds)

```Python
from sklearn.model_selection import cross_val_score
reg = linear_model.LinearRegression()
cv_results = cross_val_score(reg,X,y,cv=5)
print(cv_results)

```


### Exercice 9: 5-fold cross-validation

```Python
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
```

### Exercice 10: K-Fold CV comparison

```Python
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg,X,y,cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg,X,y,cv=10)
print(np.mean(cvscores_10))

```

## Regularized regression

* Why regularize ?
  * Linear regression minimizes a loss function
  * It chooses coefficient for each feature variable
  * Large coefficients can lead to overfitting
  * Penalizing large coefficients : Regularization
* Ridge regression
  * Loss function = 

OLS Loss function +
$$\alpha*\sum_{i=1}^{n}a_{i}^2$$

* alpha : parameter we need to choose
* Picking alpha is similar to picking k in k-NN
  * It's called HyperParameter tuning
* Alpha controls model complexity
  * Alpha = 0: Weg back OLS (can lead to overfitting)
  * Very high Alpha: can lead to underfitting
  * Ridge is L2 regularisation = we penalize by the square of the parameter

```Python
from sklearn.linear_model from Ridge
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
ridge = Ridge(alpha=0.1,normalized=True)
ridge.fit(X_train,y_train)A
ridge_pred = ridge.predict(X_test)
ridge.score(X_test,y_test)

```

* Lasso regression (L1 Regularisation)
  * Loss function = OLS loss function +
    $$ \alpha*\sum_{i=1}^{n}\lvert a_i \lvert$$

```Python
from sklearn.linear_model import Lasso
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
lasso = Lasso(alpha=0.1,normalized=True)
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test,y_test)
```
* Lasso regression can be used to select important features of a dataset
* Shrinks the coefficients of less important features to exactly 0 
* Lasso is good tool for features selection but ridge is better for prediction

```Python
from sklearn.linear_model import Lasso
names = boston.drop('MEDV',axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef= lasso.fit(X,y).coef_
_ = plt.plot(range(len(names)),lasso_coef)
_ = plt.xticks(range(len(names)),names,rotation=60)
_ = plt.ylabel('Coefficients')
```

#### Exercice 11 : calculation of the features that influence the result of a predicted value

```Python
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4,normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
```

#### Exercice 12 : Rigde linear regression

```Python
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

 # Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
```