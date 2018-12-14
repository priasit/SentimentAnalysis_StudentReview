import re
import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import sklearn
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

#
# train - training data, with an additional column called 'Label' for learning. 
#	1 - is positive sentiment (referred to as 'regular' in the code)
#	3 - is negative sentiment (referred to as 'negative' in the code)
#
# test - the test data, that would be combined with the training data
#
train  = pd.read_csv('TrainingHSReviews.csv')

count = len(sys.argv)
if count != 2:
	print ("Need 1 review file as arguments to process");
	sys.exit(0)

print("Processing test file: ", sys.argv[1])
test = pd.read_csv(sys.argv[1])

#test = pd.read_csv('PaloAltoHSReviews.csv') 
#test = pd.read_csv('TestHSReviews.csv')


#
# Just to see what is in the trainning data set - just the first few rows. 
#
train.head()

#
# Combining testing and training data to avoid double work
# Note: ignore_index is set to true as there are no index columns
#
combi = train.append(test, ignore_index=True)

#
# A function to remove unwanted characters in the Sentiment text
#
def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)
        
	return input_txt 
	
#
# Another way to remove unwanted characters
# In this we are replacing all characters that are not lower case a-z, upper case A-Z, and 0-9, with a space
#
combi['cleaned_sentiment'] = combi['Sentiment'].str.replace("[^a-zA-Z0-9]", " ")

#
# Another way to remove unwanted characters
# In this we are removing all the words having length 3 or less - they may not have any meaning. ex: expressions such as "uhh"
#
combi['cleaned_sentiment'] = combi['cleaned_sentiment'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#
# Just to see what is in the combined data set - just the first few rows. 
#
combi.head()

#
# Tokneinizing the Sentiments with a lambda function.
#
tokenized_sentiment = combi['cleaned_sentiment'].apply(lambda x: x.split())

#
# Just to see what is in the Sentiment tokens data set - just the first few rows. 
#
tokenized_sentiment.head()

#
# Rule-based process of stripping meaningless suffixes with stemming. Ex: Connect, Connection, Connected all
# have Connect as the root.
#
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_sentiment = tokenized_sentiment.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
#
# Just to see what is in the stemmed Sentiment tokens data set - just the first few rows. 
#
tokenized_sentiment.head()

#
# Now stitching the tokens together with spaces in between.
#
for i in range(len(tokenized_sentiment)):
    tokenized_sentiment[i] = ' '.join(tokenized_sentiment[i]) #

#
# The combined list of tokecns now contains clean, stemmed, and stitched tokens.
#
combi['cleaned_sentiment'] = tokenized_sentiment

#
# Plot the list of sanitized tokens.
# Trying to learn  but did not use .
#
all_words = ' '.join([text for text in combi['cleaned_sentiment']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
#plt.show()

#
# Plot the list of sanitized tokens.
#  Trying to learn.
# This plot if for positive (Label value = 1) sentiment.
#
normal_words =' '.join([text for text in combi['cleaned_sentiment'][combi['Label'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
#plt.show()

#
# Plot the list of sanitized tokens.
# Trying to learn 
# This plot if for negative (Label value = 3) sentiment.
#
negative_words = ' '.join([text for text in combi['cleaned_sentiment'][combi['Label'] == 3]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
#plt.show()

# function to collect sentimentwords
def sentimentword_extract(x):
    sentimentwords = []
    # Loop over the words in the Sentiment.
    for i in x:
        ht = re.findall(r"(\w+)", i)
        sentimentwords.append(ht)

    return sentimentwords

#	
# Extracting sentimentwords from positive (Label value = 1)Sentiments.
#
sentiment_regular = sentimentword_extract(combi['cleaned_sentiment'][combi['Label'] == 1])

#	
# Extracting sentimentwords from positive (Label value = 3)Sentiments.
#
sentiment_negative = sentimentword_extract(combi['cleaned_sentiment'][combi['Label'] == 3])

#
# Separating the Sentiment in two separate lists.
# Regular - Positive Sentiment.
# Negative - Negative Sentiment.
#
sentiment_regular = sum(sentiment_regular,[])
sentiment_negative = sum(sentiment_negative,[])

#
# Get the data from the Pandas DataFrame structure.
# Selecting top 10 most frequent positive sentimentwords.
# Plot the graph.
#
a = nltk.FreqDist(sentiment_regular)
d = pd.DataFrame({'Sentiment': list(a.keys()),
                  'Count': list(a.values())})
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Sentiment", y = "Count")
ax.set(ylabel = 'Count')
#plt.show()

#
# Get the data from the Pandas DataFrame structure.
# Selecting top 10 most frequent negative sentimentwords.
# Plot the graph.
# Trying to learn but did not print

b = nltk.FreqDist(sentiment_negative)
e = pd.DataFrame({'Sentiment': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent words
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Sentiment", y = "Count")
ax.set(ylabel = 'Count')
#plt.show()

#
# Tokenizer to remove unwanted elements from the data like symbols and numbers, once more
#
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(combi['cleaned_sentiment'])

#
# To understand model performance, dividing the dataset into a training set and a test set 
# Splitting dataset by using function train_test_split().  
# Passing 3 parameters features, target, and test_set size.  
# use random_state to select records randomly.
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, combi['cleaned_sentiment'], test_size=0.3, random_state=1)

#
# Building the Text Classification Model using TF-IDF. Importing the MultinomialNB module and creating
# a Multinomial Naive Bayes classifier object using MultinomialNB() function. Fitting the model on a training 
# set using fit() and performing prediction on the test set using predict().
#	
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("Naive Bayes MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(combi['cleaned_sentiment'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, combi['cleaned_sentiment'], test_size=0.3, random_state=123)

#
# Model Building and Evaluation (TF-IDF). Building the Text Classification Model using TF-IDF.
	
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("TF-IDF MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
