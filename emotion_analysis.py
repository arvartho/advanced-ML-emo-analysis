# -*- coding: utf-8 -*-
"""Emotion Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1op6LbKx8sgz2cnHbHJxSYUI2shk0f5-n
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from textblob import TextBlob

from datetime import datetime
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter

from TweetProcessor import TweetProcessor as tp
tweet_processor = tp.TweetProcessor()
start_all = datetime.now()

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(y_test, y_pred):
   accuracy = np.around(metrics.accuracy_score(y_test, y_pred),3)
   precision = np.around(metrics.precision_score(y_test, y_pred, average='macro'),3)
   recall = np.around(metrics.recall_score(y_test, y_pred, average='macro'),3)
   f1_score = np.around(metrics.f1_score(y_test, y_pred, average='macro'),3)
   print(classification_report(y_test, y_pred))  
   print('accuracy: %s' % accuracy)
   print('precision: %s' % precision)
   print('recall: %s' % recall)
   print('f1_score: %s' % f1_score)
   return {'accuracy':accuracy, 
           'precision':precision, 
           'recall':recall,
           'f1_score':f1_score}

def textblob_sentiment(doc):
   from textblob import TextBlob
   score = TextBlob(doc).sentiment.polarity
   if score == 0:
      return 'neutral'
   elif score < 0:
      return 'negative'
   else:
      return 'positive'
   
def plot_confusion_matrix(y_test, y_pred):
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.metrics import classification_report, confusion_matrix
   
   labels = np.unique(y_test)
   cm = confusion_matrix(y_test, y_pred, labels=labels)
   cm_df = pd.DataFrame(cm, index=labels, columns=labels)
   print(cm_df)
   cmap = sns.color_palette("OrRd", 1000)
   ax = sns.heatmap(cm_df, cmap=cmap, annot=True, fmt='g')
   bottom, top = ax.get_ylim()
   ax.set_ylim(bottom + 0.5, top - 0.5)
   
def vizualize_classification(y_test, y_pred):
   from sklearn.metrics import classification_report, confusion_matrix
   labels = np.unique(y_test)
   cmap = sns.color_palette("OrRd", 1000)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,7))
   fig.suptitle('Vizualizing classifier')
   
   ax = sns.countplot(x=y_pred, hue=y_test, order=labels, ax=ax1)
   
   cm = confusion_matrix(y_test, y_pred, labels=labels)
   cm_df = pd.DataFrame(cm, index=labels, columns=labels)
   print('Confusion Matrix: \n', cm_df)
   ax = sns.heatmap(cm_df, cmap=cmap, annot=True, fmt='g', ax=ax2)
   bottom, top = ax.get_ylim()
   ax.set_ylim(bottom + 0.5, top - 0.5)
   evaluate_model(y_test, y_pred)
      
def textblob_NBClassifier(doc):
   blob = TextBlob(doc, classifier=cl)
   return blob.classify()

def multilabel_visualization(values, labels):
   df = pd.DataFrame(data={label:values[:,i] for i, label in enumerate(feelings)})
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13,7))
   fig.suptitle('Class analysis')
   # Label distribution
   df[labels].sum().sort_values(ascending=False).plot(kind='bar', title='Label distribution', ax=ax1)
   # Multiple labels
   sns.countplot(x=df[labels].sum(axis=1), ax=ax2).set_title('Tweets with multiple labels')
   # Word cloud
   # Create target column 'labels'
   df['labels'] = [np.asarray(labels)[i] for i in df[labels].astype('bool').values]
   # Concatenate all labesl together
   words = np.hstack(df['labels'].values)
   # Vizualize label wordcloud
   wc_dict = dict(Counter(words))
   wordcloud = WordCloud(colormap='Blues', 
                         max_font_size=100, 
                         width=300, 
                         height=200).generate_from_frequencies(wc_dict)
   plt.title('Label wordcloud')
   # plt.figure( figsize=(20,10) )

   plt.imshow(wordcloud, interpolation='bilinear')
   plt.axis("off")
   plt.show()
   # correlation visualization
   emotion_corr_heatmap(values, labels=labels)
   
def emotion_corr_heatmap(values, labels=[]):
   if type(values)==pd.DataFrame:
      data = values.corr()
   else:
      data = pd.DataFrame(data=values, columns=labels).corr()
   
   # Generate a mask for the upper triangle
   mask = np.triu(np.ones_like(data, dtype=np.bool))
   cmap = sns.diverging_palette(220, 10, as_cmap=True)
   # Draw the heatmap with the mask and correct aspect ratio
   max_val = np.sort(data, axis=None)[::-1][data.shape[1]]
   ax = sns.heatmap(data, 
               mask=mask, 
               cmap=cmap, 
               center=0,
               square=True, 
               linewidths=.5, 
               cbar_kws={"shrink": .5})
   plt.title('Primitive emotion correlation')
   bottom, top = ax.get_ylim()
   ax.set_ylim(bottom + 0.5, top - 0.5)
   
def draw_wordcloud(corpus):
   from wordcloud import WordCloud, STOPWORDS
   from collections import Counter
   from nltk.util import ngrams 
   from nltk.corpus import stopwords
   
   if type(corpus)!=str:
      corpus = ' '.join(corpus).split(' ')
   stop_words = set(list(STOPWORDS) + stopwords.words('english'))
   corpus = [w for w in corpus if w.lower() not in stop_words]
   bigrams_count = {' '.join(ngram):count for ngram,count in Counter(ngrams(corpus, 2)).most_common()}
   trigrams_count = {' '.join(ngram):count for ngram,count in Counter(ngrams(corpus, 3)).most_common()}   
   corpus_dict = dict(Counter(corpus).most_common())
   word_count = {**corpus_dict, **bigrams_count, **trigrams_count}
   # draw a Word Cloud with word frequencies
   wordcloud = WordCloud(max_words=50,
                         max_font_size=75,
                         colormap='Pastel2_r',
                        ).generate_from_frequencies(word_count)
   plt.figure(figsize=(10,8))
   plt.title('Wordcloud of unigrams, bigrams, trigrams')
   plt.imshow(wordcloud, interpolation='bilinear')
   plt.axis("off")
   plt.show()

def addValuesToResults(result):
   
   accuracy_res.append(result['accuracy'])
   precision_res.append(result['precision'])
   recall_res.append(result['recall'])
   f1_res.append(result['f1_score'])
   hamming_res.append(result['hamming_loss'])

emo_train_df = pd.read_csv('data/2018-E-c-En-train.txt', sep='\t')
emo_test_df = pd.read_csv('data/2018-E-c-En-test-gold.txt', sep='\t')
print(emo_train_df.shape)
emo_train_df.head()

"""## Data exploration

Class distribution
"""

feelings = emo_train_df.columns[2:]#['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
multilabel_visualization(emo_train_df[feelings].values, feelings)

"""Preprocessing:
* Tokenization
* Stopword removal
* Special character handling
* Emoji parsing
* Data cleaning
"""

emo_train_df['tokenized_features'] = emo_train_df['Tweet'].apply(tweet_processor.tweet_pipeline)

corpus = emo_train_df['tokenized_features'].apply(lambda x: ' '.join(x))
draw_wordcloud(corpus)

"""## Feature selection"""

# Tweet preprocessing
emo_train_df['tokenized_features'] = emo_train_df['Tweet'].apply(tweet_processor.tweet_pipeline)
X = emo_train_df['tokenized_features'].apply(lambda x: ' '.join(x)).values

# TF-IDF text features
vectorizer = tweet_processor.tfidf_vectorizer(X)
X_tfidf = vectorizer.transform(X)

# TF-IDF emoji features
vectorizer = tweet_processor.emoji_vectorizer(X)
X_emoji = vectorizer.transform(X)

# LDA features
X_lda = tweet_processor.lda_features(X, len(feelings))

"""Dataset split"""

# Combine features
X_combined = np.hstack((X_tfidf.toarray(), X_emoji.toarray()))
y = emo_train_df[feelings].values
x_train_id, x_test_id, y_train, y_test = train_test_split(range(X_combined.shape[0]), y, test_size=0.2, random_state=0)
x_train, x_test = X_combined[x_train_id], X_combined[x_test_id]
x_train.shape, y_train.shape, x_test.shape

from MultiLabel.MultiLabel import Multilabel

Multilabel = Multilabel(x_train, y_train, x_test, y_test)

accuracy_res = []
precision_res = []
recall_res = []
f1_res = []
hamming_res = []
labels = []

results = Multilabel.oneVsRest()
addValuesToResults(results)
labels.append("oneVsRest")

#results = Multilabel.MLkNN()
#addValuesToResults(results)
#labels.append("MLkNN")
#
results = Multilabel.LabelPowerset()
addValuesToResults(results)
labels.append("LabelPowerset")
#
#results = Multilabel.ClassifierChain()
#addValuesToResults(results)
#labels.append("ClassifierChain")
#
#results = Multilabel.BinaryRelevance()
#addValuesToResults(results)
#labels.append("BinaryRelevance")

results = Multilabel.MajorityVotingClassifier()
addValuesToResults(results)
labels.append("MajVotClf")

Multilabel.plotScores(accuracy_res, precision_res, recall_res, f1_res, hamming_res, labels)
