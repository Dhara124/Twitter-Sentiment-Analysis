#!/usr/bin/env python
# coding: utf-8

# # TASK #1: INTRODUCTION TO GUIDED PROJECT 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# data source: https://www.kaggle.com/sid321axn/amazon-alexa-reviews/kernels

# # TASK #2: PROJECT WALKTHROUGH, ENHANCED FEATURES, AND LEARNING OUTCOMES 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # PRACTICE OPPORTUNITY #1: 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #3: IMPORT LIBRARIES AND DATASETS

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[2]:


# Load the data
tweets_df = pd.read_csv('twitter.csv')


# In[3]:


tweets_df


# In[4]:


tweets_df.info()


# In[5]:


tweets_df.describe()


# In[6]:


tweets_df['tweet']


# In[7]:


# Drop the 'id' column
tweets_df = tweets_df.drop(['id'], axis=1)


# # TASK #3: PERFORM DATA EXPLORATION 

# In[8]:


sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[9]:


tweets_df.hist(bins = 30, figsize = (13,5), color = 'r')


# In[10]:


sns.countplot(tweets_df['label'], label = "Count") 


# In[11]:


# Let's get the length of the messages
tweets_df['length'] = tweets_df['tweet'].apply(len)


# In[12]:


tweets_df


# In[13]:


tweets_df.describe()


# In[14]:


# Let's see the shortest message 
tweets_df[tweets_df['length'] == 11]['tweet'].iloc[0]


# # PRACTICE OPPORTUNITY #2: 

# ![image.png](attachment:image.png)

# In[15]:


# Let's view the message with mean length 
tweets_df[tweets_df['length'] == 84]['tweet'].iloc[0]


# In[16]:


# Plot the histogram of the length column
tweets_df['length'].plot(bins=100, kind='hist') 


# # TASK #4: PLOT THE WORDCLOUD

# In[17]:


positive = tweets_df[tweets_df['label']==0]
positive


# In[18]:


negative = tweets_df[tweets_df['label']==1]
negative


# In[19]:


sentences = tweets_df['tweet'].tolist()
len(sentences)


# In[20]:


sentences_as_one_string =" ".join(sentences)


# In[21]:


sentences_as_one_string


# In[22]:


get_ipython().system('pip install wordcloud')


# In[23]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# # PRACTICE OPPORTUNITY #3: 

# ![image.png](attachment:image.png)

# In[24]:


negative_list = negative['tweet'].tolist()
negative_list
negative_sentences_as_one_string = " ".join(negative_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))


# # TASK #5: PERFORM DATA CLEANING - REMOVE PUNCTUATION FROM TEXT

# In[25]:


import string
string.punctuation


# In[26]:


Test = '$I love AI & Machine learning!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[27]:


Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'


# In[28]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[29]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# # TASK 6: PERFORM DATA CLEANING - REMOVE STOPWORDS

# In[30]:


import nltk # Natural Language tool kit 
nltk.download('stopwords')

# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# In[31]:


Test_punc_removed_join = 'I enjoy coding, programming and Artificial intelligence'
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[32]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# In[33]:


Test_punc_removed_join


# # PRACTICE OPPORTUNITY #4: 

# ![image.png](attachment:image.png)

# In[34]:


mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'


# In[35]:


# Remove punctuations
challege = [ char     for char in mini_challenge  if char not in string.punctuation ]
challenge = ''.join(challege)
challenge


# In[36]:


challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 
challenge


# # TASK 7: PERFORM COUNT VECTORIZATION (TOKENIZATION)

# ![image.png](attachment:image.png)

# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[38]:


print(vectorizer.get_feature_names())


# In[39]:


print(X.toarray())  


# # PRACTICE OPPORTUNITY #5: 

# ![image.png](attachment:image.png)

# In[40]:


mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

# mini_challenge = ['Hello World', 'Hello Hello Hello World world', 'Hello Hello World world world World']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())



# # TASK #8: CREATE A PIPELINE TO REMOVE PUNCTUATIONS, STOPWORDS AND PERFORM COUNT VECTORIZATION

# In[41]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[42]:


# Let's test the newly added function
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)


# In[43]:


print(tweets_df_clean[5]) # show the cleaned up version


# In[44]:


print(tweets_df['tweet'][5]) # show the original version


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])


# In[47]:


print(vectorizer.get_feature_names())


# In[48]:


print(tweets_countvectorizer.toarray())  


# In[49]:


tweets_countvectorizer.shape


# In[50]:


X = pd.DataFrame(tweets_countvectorizer.toarray())


# In[51]:


X


# In[52]:


y = tweets_df['label']


# # TASK #9: UNDERSTAND THE THEORY AND INTUITION BEHIND NAIVE BAYES

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # PRACTICE OPPORTUNITY #6: 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #10: TRAIN AND EVALUATE A NAIVE BAYES CLASSIFIER MODEL

# In[53]:


X.shape


# In[54]:


y.shape


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[56]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# ![image.png](attachment:image.png)

# In[57]:


from sklearn.metrics import classification_report, confusion_matrix


# In[58]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[59]:


print(classification_report(y_test, y_predict_test))


# # FINAL PROJECT

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # FINAL PROJECT SOLUTION TASK #1: IMPORT DATA AND PERFORM EXPLORATORY DATA ANALYSIS 

# In[60]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[61]:


# Load the data
reviews_df = pd.read_csv('amazon_reviews.csv')
reviews_df


# In[62]:


# View the DataFrame Information
reviews_df.info()


# In[63]:


# View DataFrame Statistical Summary
reviews_df.describe()


# In[64]:


# Plot the count plot for the ratings
sns.countplot(x = reviews_df['rating']) 


# In[65]:


# Let's get the length of the verified_reviews column
reviews_df['length'] = reviews_df['verified_reviews'].apply(len)


# In[66]:


reviews_df


# In[67]:


# Plot the histogram for the length
reviews_df['length'].plot(bins=100, kind='hist') 


# In[68]:


# Apply the describe method to get statistical summary
reviews_df.describe()


# In[69]:


# Plot the countplot for feedback
# Positive ~2800
# Negative ~250
sns.countplot(x = reviews_df['feedback'])


# # FINAL PROJECT SOLUTION TASK #2: PLOT WORDCLOUD

# In[70]:


# Obtain only the positive reviews
positive = reviews_df[reviews_df['feedback'] == 1]
positive


# In[71]:


# Obtain the negative reviews only
negative = reviews_df[reviews_df['feedback'] == 0]
negative


# In[72]:


# Convert to list format
sentences = positive['verified_reviews'].tolist()
len(sentences)


# In[73]:


# Join all reviews into one large string
sentences_as_one_string =" ".join(sentences)


# In[74]:


sentences_as_one_string


# In[75]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[ ]:


sentences = negative['verified_reviews'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize = (20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# # FINAL PROJECT SOLUTION TASK #3: PERFORM DATA CLEANING 

# In[ ]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[ ]:


# Let's test the newly added function
reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)


# In[ ]:


# show the original review
print(reviews_df['verified_reviews'][5]) 


# In[ ]:


# show the cleaned up version
print(reviews_df_clean[5])


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])


# In[ ]:


print(vectorizer.get_feature_names())


# In[ ]:


print(reviews_countvectorizer.toarray())  


# In[ ]:


reviews_countvectorizer.shape


# In[ ]:


reviews = pd.DataFrame(reviews_countvectorizer.toarray())


# In[ ]:


X = reviews


# In[ ]:


y = reviews_df['feedback']
y


# # FINAL PROJECT SOLUTION TASK #4: TRAIN AND TEST AI/ML MODELS

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[ ]:


print(classification_report(y_test, y_predict_test))


# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))


# # EXCELLENT JOB! YOU SHOULD BE PROUD OF YOUR NEWLY ACQUIRED SKILLS

# In[ ]:




