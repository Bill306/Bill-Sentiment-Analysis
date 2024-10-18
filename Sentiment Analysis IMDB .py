#!/usr/bin/env python
# coding: utf-8

# # Step 1: Problem Definition

# "IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
# This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms."

# Essentially, we are: classifing the sentiment of movie reviews as either positive or negative using the IMDB dataset

# # Step 2: Data

# Looking at the dataset from Kaggle, the IMDB Dataset.csv contains two columns, the review and and sentified classification(positive or negative). 
# 
# The IMDB Dataset.csv has 50K movie review. 
# 

# # Step 3: Evaluation

# The page does not specify any evaluation matrix. Since it's a NLP Challenge, we can use: 
# - Accuracy: The proportion of correctly classified reviews, which gives a basic measure of overall performance.
# - Precision: Evaluate how many of the reviews predicted as positive are actually positive.
# - Recall: Measures the model's ability to identify all positive reviews correctly.
# - F1 Score: The harmonic mean of precision and recall, useful when both false positives and false negatives are equally important to minimize.
# - Confusion Matrix: This provides a detailed breakdown of correct and incorrect predictions for both positive and negative reviews, offering deeper insights into the performance of the model.
# - AUC-ROC (Area Under the Receiver Operating Characteristic Curve): Useful for evaluating the model’s ability to distinguish between positive and negative reviews, especially when considering different classification thresholds.

# # Step 4: Features

# The data contains two features: 
# 1. Text Data (Movie Reviews):
# The main feature of this dataset is the review text itself, which contains the actual words, sentences, and paragraphs written by users about the movies.
# 
# 2. Sentiment  Classification： 
# The sentiment classification categorizes each review into either 'positive' or 'negative'. 

# # Step 5: Modeling 

# I use logisticregression in this case because it good for binary classification problems like sentiment analysis. It handles large, sparse data efficiently, avoids overfitting, provides probabilistic outputs, and is fast to train and predict. 
# Additionally, it performs well in linearly separable tasks, making it an ideal choice for this sentiment classification problem.

# # Code Section

# In[8]:


#Import all the necessary library

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re


# ### Understanding the Data 

# In[2]:


df = pd.read_csv('IMDB Dataset.csv')

df


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


# Checking Null Value
df.isnull()


# ## Data Cleaning 

# In[9]:


#Text Preprocessing - Clean the review text
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<br />', ' ', text)
    # Remove non-alphabetical characters and lowercase the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

df['cleaned_review'] = df['review'].apply(preprocess_text)


# In[10]:


df['cleaned_review']


# In[11]:


# Convert 'sentiment' to numeric (1 for positive, 0 for negative)
label_encoder = LabelEncoder()
df['sentiment_label'] = label_encoder.fit_transform(df['sentiment'])


# In[12]:


df


# ### Data Spliting 

# In[13]:


#Split the dataset into training and testing sets
X = df['cleaned_review']
y = df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Feature Extraction Using  using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# I use TF-IDF in this case to convert the movie reviews (text) into numerical data so that  machine learning models can process. 
# TF-IDF can: 
# - Transforms text into numerical features, allowing models to work with the data.
# - Emphasizes important words (like "amazing," "horrible") that are key for sentiment analysis.
# - Down-weights common words (like "the," "and") that are less meaningful for classification.
# - Balances word frequency, ensuring that words rare across the dataset but frequent in a review get more weight.
# - Prevents bias toward longer reviews and improves performance by focusing on sentiment-related terms.

# ### Model Training and Predictions

# In[15]:


#Train a Logistic Regression classifier
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_tfidf, y_train)


# In[16]:


# Make predictions on the test set
y_pred = lr_model.predict(X_test_tfidf)


# ### Model Evaluation

# In[17]:


#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[18]:


# Display the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: \n{conf_matrix}")


# The model performs well in classifying both positive and negative reviews, with a good balance between precision (88.8%) and recall (90.6%). The confusion matrix shows that it makes a moderate number of errors (578 false positives and 472 false negatives), but the overall performance (accuracy of 89.5%) is strong.
# 
# Precision indicates the model is reliable in its positive predictions, and recall shows it effectively identifies positive reviews. The F1 score further confirms the balance between precision and recall.

# 
