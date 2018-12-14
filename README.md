# SentimentAnalysis_StudentReview
Sentiment Analysis of Student Review
The code changes for this project were implemented in Python using NLTK
The process involved: 
1. Gathering data and putting it in a .CSV file format. 
2. Cleaning up the Data. 
3. Performing Sentiment Analysis 
4. Determining the score. 
Took data set from niche.com—Downloaded the reviews for each school as a.txt file. 
 
 Wrote a python program to get it into a CSV file. 
For creating the training data set, created the program readTrainingDataV11.py to create the “TrainingHSReviews.CSV” file . Added a column ‘Label’ to establish sentiment for learning. 
1- Positive 
2- Neutral 
3- Negative 
 For this project  only used  Positive and Negative 
Created the test data set  “TestHSReviews.CSV” by writing the program  readTestData.py 

Wrote the SentimentAnalysisFinal.py program to perform sentiment analysis on the input test data and it involved the following: 
For data cleanup 
1. I replaced all characters that are not lower case a-z, upper case A-Z, and 0-9, with a space. 
2. Removed all the words having length 3 or less  as  they may not have any meaning. Example : "uhh“ 
Created a list of positive and negative sentiment words. 
Used the NLTK for tokenization. Used  PorterStemmer for Stemming. 
To understand model performance, divided the dataset into a training set and a test set . 
Built the Text Classification Model using TF-IDF. Importing the MultinomialNB module and creating a Multinomial Naive Bayes classifier object using MultinomialNB() function. Fitting the model on the training set using fit() and performing prediction on the test set using predict(). 
