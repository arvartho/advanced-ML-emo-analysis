# -*- coding: utf-8 -*-

from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.metrics import hamming_loss, multilabel_confusion_matrix
import numpy as np
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, BinaryRelevance
from sklearn.svm import SVC
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

class Multilabel:
   def __init__(self, x_train, y_train, x_test, y_test):
      self.x_train = x_train
      self.y_train = y_train
      self.x_test = x_test
      self.y_test = y_test
   
   def plotScores(self,accuracy_res, precision_res, recall_res, f1_res, hamming_res, labels):
      fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize=(16, 5))

      x = np.arange(len(accuracy_res))
      
      xlabels = list(labels)
      sns.barplot(x, accuracy_res, palette='Blues', ax=ax1)
      sns.barplot(x, precision_res, palette='Reds', ax=ax2)
      sns.barplot(x, recall_res, palette='Greens', ax=ax3)
      sns.barplot(x, f1_res, palette='RdBu_r', ax=ax4)
      sns.barplot(x, hamming_res, palette='coolwarm', ax=ax5)
      
      ax1.set_ylabel('Accuracy')
      ax2.set_ylabel('precision')
      ax3.set_ylabel('Recall')
      ax4.set_ylabel('F1-Score')
      ax5.set_ylabel('Hamming Loss')
      
      # # Add the xlabels to the chart
      ax1.set_xticklabels(xlabels)
      ax2.set_xticklabels(xlabels)
      ax3.set_xticklabels(xlabels)
      ax4.set_xticklabels(xlabels)
      ax5.set_xticklabels(xlabels)
      
      # Add the actual value on top of each bar
      for i, v in enumerate(zip(accuracy_res, precision_res, recall_res,f1_res,hamming_res)):
         ax1.text(i - 0.1, v[0] + 0.01, str(round(v[0], 2)))
         ax2.text(i - 0.1, v[1] + 0.01, str(round(v[1], 2)))
         ax3.text(i - 0.1, v[2] + 0.01, str(round(v[2], 2)))
         ax4.text(i - 0.1, v[2] + 0.01, str(round(v[2], 2)))
         ax5.text(i - 0.1, v[2] + 0.01, str(round(v[2], 2)))
      
      # Show the final plot
      plt.show()

   
   def multilabel_evaluation(self, y_pred, y_test, measurement=None):
      '''
      Micro accuracy, recall, precision, f1_score evaluation
      '''
      
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

   def oneVsRest(self):
      print("")
      print("Starting One Vs Rest Classifier of sklearn.multiclass...")
      print("")
      start = datetime.now()
      
      pipeline = Pipeline([
                     ('clf', OneVsRestClassifier(BernoulliNB(class_prior=None)))
                  ])
      
      parameters = [{
                  'clf__estimator__alpha': (0.5, 0.7, 1),
                  }]
      
      grid_search_cv = GridSearchCV(pipeline, 
                                    parameters, 
                                    cv=2,
                                    verbose=2,
                                    n_jobs=-1)
      grid_search_cv.fit(self.x_train, self.y_train)
      best_clf = grid_search_cv.best_estimator_
      
      print('Finished training in : ', datetime.now()-start) 
      
      y_pred = best_clf.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)

   def oneVsone(self):
      print("")
      print("Starting One Vs One Classifier of sklearn.multiclass...")
      print("")
      start = datetime.now()
      
      pipeline = Pipeline([
                     ('clf', OneVsOneClassifier(BernoulliNB(class_prior=None)))
                  ])
      
      parameters = [{
                  'clf__estimator__alpha': (0.5, 0.7, 1),
                  }]
      
      grid_search_cv = GridSearchCV(pipeline, 
                                    parameters, 
                                    cv=2,
                                    verbose=2,
                                    n_jobs=-1)
      grid_search_cv.fit(self.x_train, self.y_train)
      best_clf = grid_search_cv.best_estimator_
      
      print('Finished training in : ', datetime.now()-start) 
      
      y_pred = best_clf.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)
   
   def MLkNN(self):
      print("")
      print("Starting MLkNN Classifier of skmultilearn.adapt...")
      print("")
      start = datetime.now()
      
      parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
      
      grid_search_cv = GridSearchCV(MLkNN(), parameters, scoring='f1_macro',
                                    verbose=2,
                                    n_jobs=-1)
      grid_search_cv.fit(self.x_train, self.y_train)
      best_clf = grid_search_cv.best_estimator_
      
      print('Finished training in : ', datetime.now()-start) 
      
      y_pred = best_clf.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)
   
   def LabelPowerset(self):
      print("")
      print("Starting LabelPowerset Classifier of skmultilearn.problem_transform...")
      print("")
      start = datetime.now()
      
      parameters = [
      {
         'classifier': [MultinomialNB()],
         'classifier__alpha': [0.7, 1.0],
      },
      {
         'classifier': [RandomForestClassifier()],
         'classifier__criterion': ['gini', 'entropy'],
         'classifier__n_estimators': [10, 20, 50],
      }
      ]
      
      grid_search_cv = GridSearchCV(LabelPowerset(), parameters, scoring='f1_macro',
                                    verbose=2,
                                    n_jobs=-1)
      grid_search_cv.fit(self.x_train, self.y_train)
      best_clf = grid_search_cv.best_estimator_
      
      print('Finished training in : ', datetime.now()-start) 
      
      y_pred = best_clf.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)
   
   def ClassifierChain(self):
      print("")
      print("Starting ClassifierChain Classifier of skmultilearn.problem_transform...")
      print("")
      start = datetime.now()
      
      parameters = [
         {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.7, 1.0],
         },
         {
            'classifier': [SVC()],
            'classifier__kernel': ['rbf', 'linear'],
         },
      ]
      
      grid_search_cv = GridSearchCV(ClassifierChain(), parameters, scoring='f1_macro',
                                    verbose=2,
                                    n_jobs=-1)
      grid_search_cv.fit(self.x_train, self.y_train)
      best_clf = grid_search_cv.best_estimator_
      
      print('Finished training in : ', datetime.now()-start) 
      
      y_pred = best_clf.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)

   def BinaryRelevance(self):
      print("")
      print("Starting BinaryRelevance Classifier of skmultilearn.problem_transform...")
      print("")
      start = datetime.now()
      
      parameters = [
         {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.7, 1.0],
         },
         {
            'classifier': [SVC()],
            'classifier__kernel': ['rbf', 'linear'],
         },
      ]
      
      grid_search_cv = GridSearchCV(BinaryRelevance(), parameters, scoring='f1_macro',
                                    verbose=2,
                                    n_jobs=-1)
      grid_search_cv.fit(self.x_train, self.y_train)
      best_clf = grid_search_cv.best_estimator_
      
      print('Finished training in : ', datetime.now()-start) 
      
      y_pred = best_clf.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)

   def MajorityVotingClassifier(self):
      print("")
      print("Starting MajorityVotingClassifier Classifier of skmultilearn.ensemble...")
      print("")
      start = datetime.now()
      
      classifier = MajorityVotingClassifier(
         clusterer = FixedLabelSpaceClusterer(clusters = [[1,2,3], [0, 2, 5], [4, 5]]),
         classifier = ClassifierChain(classifier=GaussianNB())
      )
      
      print('Finished training in : ', datetime.now()-start) 
      
      classifier.fit(self.x_train, self.y_train)
      y_pred = classifier.predict(self.x_test)
      start = datetime.now()
      return self.multilabel_evaluation(y_pred, self.y_test)
      print('Finished classification in : ', datetime.now()-start)