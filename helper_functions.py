import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(y_test, y_pred):
   from sklearn import metrics
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

def plot_roc(y_test, proba, labels):
   from sklearn.metrics import roc_curve, auc
   colors = sns.color_palette("muted", n_colors=11)
   for i,label in enumerate(labels):
      fpr, tpr,_ = roc_curve(y_test[:, i], proba[:, i])
      roc_auc = auc(fpr, tpr)   
      plt.plot(fpr, tpr, color=colors[i],
               lw=2, label='%s ROC curve (area = %0.2f)' % (label, roc_auc))
   plt.legend(loc="lower right", bbox_to_anchor=(1.7,0))
   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('%s ' % label)
   plt.show()

def multilabel_evaluation(y_pred, y_test, measurement=None):
   '''
   Micro accuracy, recall, precision, f1_score evaluation
   '''
   from sklearn.metrics import hamming_loss, multilabel_confusion_matrix
   multilabel_cm = multilabel_confusion_matrix(y_pred, y_test)
   if measurement=='macro':
      tn = np.mean(multilabel_cm[:, 0, 0])
      tp = np.mean(multilabel_cm[:, 1, 1])
      fp = np.mean(multilabel_cm[:, 0, 1])
      fn = np.mean(multilabel_cm[:, 1, 0])
      accuracy = np.around(((tp + tn)/(tn + tp + fn +fp)), 3)
      precision = np.around((tp/(tp + fp)), 3)
      recall = np.around((tp/(tp + fn)), 3)
      f1_score = np.around(2* recall*precision/(recall + precision), 3)
   else:
      tn = multilabel_cm[:, 0, 0]
      tp = multilabel_cm[:, 1, 1]
      fp = multilabel_cm[:, 0, 1]
      fn = multilabel_cm[:, 1, 0]
      ac, p, r = [], [], []
      for i in range(len(tp)):
         ac.append((tp[i] + tn[i])/(tn[i] + tp[i] + fn[i] + fp[i]))
         p.append(0 if tp[i]==0 and fp[i]==0 else tp[i]/(tp[i] + fp[i]))
         r.append(0 if tp[i]==0 and fn[i]==0 else tp[i]/(tp[i] + fn[i]))
     
      accuracy = np.around(np.mean(ac), 3)
      precision = np.around(np.mean(p), 3)
      recall = np.around(np.mean(r), 3)   
      f1_score = np.around(2* recall*precision/(recall + precision), 3)
   hamming = np.around(hamming_loss(y_test, y_pred), 3)
   return {'accuracy':accuracy, 
           'precision':precision, 
           'recall':recall,
           'f1_score':f1_score,
           'hamming_loss':hamming}

def multilabel_visualization(values, labels):
   df = pd.DataFrame(data={label:values[:,i] for i, label in enumerate(labels)})
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13,7))
   fig.suptitle('Multilabel exploration')
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

def mean_irlbl(df, labels):
   majority_class = df[labels].sum().max()
   irlbl = majority_class/df[labels].sum()
   return np.round(np.mean(irlbl), 3)

def irlbl(df, label):
   majority_class = df[feelings].sum().max()
   irlbl = (majority_class/df[feelings].sum())[label]
   return np.round(irlbl, 3)



emo_train_df = pd.read_csv('data/2018-E-c-En-train.txt', sep='\t')
feelings = emo_train_df.columns[2:]#['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
     