# Machine Learning with the Experts: "School Budgets"

* GOAL: Create a machine learning algorithm that can automate the process
* Classfication problem
  * Pre_K
  * Reporting
  * Sharing
  * Student_type

### Exercice 1: Loading the data
```Python
import pandas as pd

df = pd.read_csv('TrainingData.csv')

df.head()

df.describe()

df.info()

```

### Exercice 2: Summarizing the data

```Python
# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
df['FTE'].dropna().hist()

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()
```

## Encoding catagories

```Python
df.label = df.label.astype('category')
```

Dummy variables

```Python
dummies = pd.get_dummies(sample_df[['label']],prefix_sep='_')
```

Encode labels as categories

```Python
categorize_label = lambda x : x.astype('category')
sample_df.label = sample_df[['label']].apply(categorize_label,axis=0) 

```

### Exercice 3: encode labels with a lambda function

```Python
# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label,axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)
```

### Exercice 4: counting unique labels
```Python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique,axis=0)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()
```

## How to measure the success of a model

* Accuracy is not suffisant (case of Spam classifier for unbalanced categories)
* Log loss
  * Measure of error

Logloss for binary classification y = {1=yes, 0=no}

* Prediction probability: p 
* N : number of rows in the dataset

$$logloss = -\frac{1}{N}\sum_{i=1}^{N}(y_ilog(p_i)+(1-y_i)log(1-p_i)) $$

```Python
import numpy as np

def compute_log_loss(predicted,actual,eps=1e-14):
  predicted = np.clip(predicted,eps,1-eps)
  loss = -1 * np.mean(actual * np.log(predicted)+(1-actual)*np.log(1 - predicted))
  return loss
```

## Building a model

* Train basic model on numeric data only
  * Want to go from raw data to prediction quickly
* Multi-class logistic regression
  * Train classifier for each label separetely and use those to predict
* Format predictions and save to CSV
* Spliting the multi-class dataset
  * Using the function train_test_split()
    * Will not work here
    * May end up with labels in test set that never appear in training set
  * Solution : StratifiedShuffleSplit
    * Only works with a single target variable
    * We have many target variables
    * multilabel_train_test_split()
      * Assure that each labels are represented in the training and the test set


Splitting the data

```Python
data_to_train = df[NUMERIC_COLUMNS].fillna(-1000)
labels_to_use = pd.get_dummies(df[LABELS])

X_train,X_test,y_train,y_test = multilabel_train_test_split(data_to_train,labels_to_ser,size=0.3,seed=123) 

```

Training the model
```Python
from sklean.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

clf = OnVsRestClassifier(LogisticRegression())  # Treats each column of y indepenently
                                                # Fits a separate classifier for each of the columns

clf.fit(X_train,y_train)

```

Exercice 5 : Split the data

```Python
# Create the new DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2, 
                                                               seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")  
print(X_test.info())
print("\ny_train info:")  
print(y_train.info())
print("\ny_test info:")  
print(y_test.info()) 
```

Exercice 6: Training a model

```Python
# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Create the DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2, 
                                                               seed=123)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))
```

## Predicting on holdout data

```Python

holdout = pd.read_csv('HoldoutData.csv',index_col=0)
holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)
predictions = clf.predict_proba(holdout)

prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS],
  prefix_sep='__').columns,
  index=holdout.index,
  data=predictions)

prediction_df.to_csv('predictions.csv')

score = score_submission(pred_path='predictions.csv')
```

### Exercie 7 : using the model for a competition

```Python
# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit it to the training data
clf.fit(X_train, y_train)

# Load the holdout data: holdout
holdout = pd.read_csv('HoldoutData.csv',index_col=0)

holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')

# Submit the predictions for scoring: score
score = score_submission(pred_path='predictions.csv')

# Print score
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))
```

## Introduction to NLP

* Data for NLP (Natural Language Processing)
  * Text, documents,speech ...
* Tokenisation
  * Splitting a string into segments
  * Store segements as list


* Bag of words representations
  * Count the number of times a word pulled out of the bag
* n-grams
  * 1-gram,2-gram, ..., n-gram


* Skit-learn tools for bag-of-words:
  * CountVectorizer()
    * Tokenizes all the strings
    * Builds a 'vocabulary'
    * Counts the occurences of each token in the vocabulary

Building a bag of words:

```Python

from sklearn.feature_extraction.text import CountVectorizer

TOKENS_BASIC = '\\S+(?=\\s+)'

df.Program_Description.fillna('',inplace=True)

vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)
```

Using CountVectorizer() on column of main dataset:

```Python

vec_basic.fit(df.Program_Description)

```

### Exercice 8 : creating a bag of words

```Python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('',inplace=True)

# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

```

### Exercice 9: Combining text columns for tokenization

```Python
# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop,axis=1)
    
    # Replace nans with blanks
    text_data.fillna("",inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)

# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)

# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))
```

## Pipelines, features & text preprocessing
```Python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

pl = Pipeline([
  ('clf',OneVsRestClassifier(LogisticRegression()))
])
```

### Exercice 10: instanciate a pipeline

```Python
# Import Pipeline
from sklearn.pipeline import Pipeline

# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Split and select numeric data only, no nans 
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)
```

### Exercice 11 : using Imputer to fill the missing data in a pipeline

```Python
# Import the Imputer object
from sklearn.preprocessing import Imputer

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=456)

# Insantiate Pipeline object: pl
pl = Pipeline([
        ('imp', Imputer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train,y_train)

# Compute and print accuracy
accuracy = pl.score(X_test,y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)
```

## Text features and features unions

How to preprocess text columns

```python
from sklearn.feature_extraction.text import CountVectorizer

X_train,X_test,y_train,y_test = train_test_split(samples_df['text'],pd.get_dummies(sample_df['label']),random_state=2)

pl = Pipeline([
  ('vec',CountVectorizer()),
  ('clf',OnVsRestClassifier(LogisticRegression()))
])

pl.fit(X_train,y_train)

accuracy = pl.score(X_test,y_test)

```

### Preprocessing with multiple dtypes

* Pipeline steps for numeric and text preprocessing can't follow each other
* Output of CountVectorize can't be input to Imputer
* Solution
  * FunctionTransformer()
  * FeatureUnion() 

#### Function Transormer

 * Turns a Python function into an object that a skikit-learn pipeline can understand
 * Need to write two functions for pipeline processing
   * Take an entire DataFrame, return numeric columns
   * Take an entire DataFrame, return text columns
* Can then preprocess numeric and text data in sepatate pipelines

```python
X_train,X_test,y_train,y_test = train_test_split(sample_df[['numeric','with_missing','text']],pd_get_dummies(sample_df['label']),random_state=2)

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

get_text_data = FunctionTransformer(lambda x: x['text'],validate=False)

get_numeric_data = FunctionTranformer(lambda x: x[['numeric','with_missing']],validate=False)
)
])
```

```python

numeric_pipeline = Pipeline([
  ('selector',get_numeric_data),
  ('imputer',Imputer())
])

text_pipeline = Pipeline([
  ('selector',get_text_data),
  ('vectorizer',CountVectorizer())
])

pl = Pipeline([
    ('union', FeatureUnion([
          ('numeric',numeric_pipeleine),
        ('text',text_pipeline)
       )
    ),
    ('clf',OneVsRestClassifier(LogistRegression()))
])
```
#### Exercice 12 : Function transformer

```python
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train,y_train)

# Compute and print accuracy
accuracy = pl.score(X_test,y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)
```
#### Exercie Exercise 13 : Multiple types of processing : FunctionTransformer

```python
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())
```

#### Exercice 14 : Putting all together

```python
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )

# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

```
## Choosing a classification model

```python

LABELS = ['Function','Use', 'Repporting']

NON_LABELS = [ c for c in df.columns if c not in LABELS]


```

Using pipeline with the main dataset

```python
import numpy as np
import pandas as pd

df = pd.read_csv('TrainingSetSample.csv',index_col=0)

dummy_labels = pd.get_dummies(df[LABELS])

X_train,X_test,y_train,y_test = multilabel_train_test_split(df[NON_LABELS],dummy_labels,0.2)

get_text_data = FunctionTransformer(combine_text_columns,validate=false)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS],validate=false)

pl = Pipeline(
  [
    (
      'union', FeatureUnion([
        ('numeric_features',Pipeline([
            ('selector',get_numeric_data),
            ('imputer',Imputer())
        ])),
        ((text_features,Pipeline([
          ('selector',get_text_data),
          ('vectorizer',CountVectorizer())
        ]))
      ])
    ),
    ('clf',OneVsRestClassifier(LogisticRegression()))
  ]
)

pl.train(X_train,y_train)

pl.score(X_Test,y_test)

```

The pipeline model is very flexible: we can change the last step of the model to see if it improves the model: Random Forest, Naïve Bayes, k-NN

### Sample:

```python

pl = Pipeline(
  [
    (
      'union', FeatureUnion([
        ('numeric_features',Pipeline([
            ('selector',get_numeric_data),
            ('imputer',Imputer())
        ])),
        ((text_features,Pipeline([
          ('selector',get_text_data),
          ('vectorizer',CountVectorizer())
        ]))
      ])
    ),
    ('clf',OneVsRestClassifier(RandomForestClassifier()))
  ]
)

```
### Exercice 15 : Using FunctionTransformer on the main dataset

```python
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2, 
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns,validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train,y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
```


```python
# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
```

```python
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=15))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
```

## Learning from the expert processing

Tricks used in machine learning competitions

* Text pre-processing
* Statiscal methods
* Computational efficiency
  
### Text preprocessing

* NLP tricks for text data
  * Tokenize on punctuation to avoid hyphens, underscore, etc ..
  *  Include unigrams and bi-grams in the model to capture important information involving multiple tokens - e.g., 'middle school'

Example: N-grams and tokenization

```python
vec = CounterVectorizer(token_pattern=TOKENS_ALPHANUEMRIC,ngram_range(1,2))
```

Submission:

```python
holdout = pd.read_csv('HoldoutData.csv',inde_col=0)

predictions = pl.predict_proba(holdout)

predictions_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS].columns),index=holdout.index,data=predictions)

predictions_df.to_csv('predictions.csv')

score = score_submission(pred_path='predictions.csv')
```

### Exercice 16 : Practice text preprocessing

```python
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])


# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
```

### Stats tricks

* Statiscal tool that the winner used: interactions terms
* Case when words are not near each others:
  * Example:
    * English teacher for 2nd grade
    * 2nd grade - budget for English teacher
  * Interactions terms mathematically describe when tokens appear together
  * Interactions terms: the math

$$ \beta_1x_1 + \beta_2x_2+\beta_3(x_1 * x_2) $$

#### Adding interaction features with scikit-learn

```python

from sklearn.preprocessing import PolynomialFeatures

interaction = PolynomialFeatures(degree=2,interraction_only=True,
  include_bias=False)

interracton.fit_transform(x)
```

* Sparce interraction features

```python
SparseInterraction(degree=2).fit_transform(x).toarray()
```

#### Exercice 17: stats tricks applied

```python
# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),  
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
```

### Computational tricks and the winning model

* Adding new features may cause enormous increase in array size
* Hashing is a way of increasing memory efficiency

When to use the hashing tricks ?
* Want to make the array as small as possible
  * Dimensinality reduction
* Particulary useful on large datasets
  * e.g., lots of text data

```python
from sklearn.feature_extraction.text import HashingVectorizer

vec = HashingVectorizer(norm=None,
      non_negative=True,
      token_pattern=TOKENS_ALPHANUMERIC,
      ngram_range=(1,2))
```   

#### Exercice 18: Implementing the hashing trick in scikit-learn

```python
# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())
```

```python
# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
```

#### Useful dataset for challenge

[https://www.datadriven.org](https://www.datadriven.org)