# -*- coding: utf-8 -*-

from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
from datetime import datetime
from sklearn.metrics import hamming_loss, multilabel_confusion_matrix
import numpy as np
from skmultilearn.adapt import MLkNN

class Multilabel:
   def __init__(self, x_train, y_train, x_test, y_test):
      self.x_train = x_train
      self.y_train = y_train
      self.x_test = x_test
      self.y_test = y_test
   
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
      print("Starting classifier...")
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
      print(self.multilabel_evaluation(y_pred, self.y_test))
      # print(best_clf.get_params())
      print('Finished classification in : ', datetime.now()-start)

   def oneVsone(self):
      print("")
      print("Starting classifier...")
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
      print(self.multilabel_evaluation(y_pred, self.y_test))
      # print(best_clf.get_params())
      print('Finished classification in : ', datetime.now()-start)
   
   def MLkNN(self):
      print("")
      print("Starting classifier...")
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
      print(self.multilabel_evaluation(y_pred, self.y_test))
      # print(best_clf.get_params())
      print('Finished classification in : ', datetime.now()-start)