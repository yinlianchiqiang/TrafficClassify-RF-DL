import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#有recall跟precision的話可以推出f1
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils.constants import ID_TO_TRAFFIC
import utils.io_util as io_util
import utils.img_util as img_util
import time

def generate_rf(train_X, train_y, val_X, val_y, tree_cnt):
  rf = RandomForestClassifier(n_estimators = tree_cnt, criterion = 'gini', random_state = 42)
  rf.fit(train_X, train_y)
  print('RF - score: ', rf.score(val_X, val_y))
  return rf

def get_pred_y(model, x, y, dataset_name):
  y_pred, acc = get_pred_acc(model, x, y)
  print(f'RF - {dataset_name} acc', acc)
  return y_pred

def get_F1_score(y_true, y_pred):
  f1 = f1_score(y_true, y_pred, average = 'micro')
  print("F1_score = ",f1)
  #return f1
  
def get_precision_score(y_true, y_pred):
  precision = precision_score(y_true, y_pred, average = 'micro')
  print("Precision_score = ",precision)

def get_recall_score(y_true, y_pred):
  recall = recall_score(y_true, y_pred, average = 'micro')
  print("Recall_score = ",recall)

def get_yPred_and_yTrue_without_one_hot(y_true, y_pred):
  #change y_pred from one_hot_encoding to normal labels
  y_pred = np.argmax(y_pred, axis=1)
  y_true = y_true.argmax(1)
  
  return y_true, y_pred

def score_for_each_class(tp,fp,fn):
  #Recall(召回率) = TP/(TP+FN)
  #Precision(準確率) = TP/(TP+FP)
  #F1-score = 2 * Precision * Recall / (Precision + Recall)
  recall = tp/(tp+fn)
  precision = tp/(tp+fp)
  f1 = 2 * precision * recall / (precision + recall)
  print("Recall : ", recall)
  print("Precision : ", precision)
  print("F1 : ", f1)

#1 看全部的分類
#2 看當中的各個類別各自的CM 

def get_TP_TN(cm):
  print(cm)
  #ecah class have there only tp,tn
  tp = np.diag(cm)
  fp = np.sum(cm, axis = 0) - tp
  fn = np.sum(cm, axis = 1) - tp
  tn = np.sum(cm) - (tp + fp + fn)
  s = np.sum(cm)
  print('tp ',tp,'\nfp ',fp,'\nfn ',fn,'\ntn ',tn) 
  for i in range(len(tp)):
    print(f"Class {i}:")
    print(f"  TP = {tp[i]/s}",end = '')
    print(f"  FP = {fp[i]/s}",end = '')
    print(f"  FN = {fn[i]/s}",end = '')
    print(f"  TN = {tn[i]/s}")
    score_for_each_class(tp[i], fp[i], fn[i])
    print('--------\n')
    #\n--------\n


  #count total 
  #FP and FN must be same
  TP = np.sum(tp)
  FP = np.sum(fp)
  FN = np.sum(fn)
  TN = np.sum(tn)
  S = np.sum([TP,FP,FN,TN])
  print("TP = ",TP/S)
  print("FP = ",FP/S)
  print("FN = ",FN/S)
  print("TN = ",TN/S)

def get_acc(model, x, y):
  y_pred, acc = get_pred_acc(model, x, y)
  return acc

def get_pred_acc(model, x, y):
  start_time = time.time() 
  
  y_pred = model.predict(x)

  end_time = time.time() 
  elapsed_time = end_time - start_time
  print(f"\n!!Execution time: {elapsed_time:.6f} seconds\n")

  return y_pred, accuracy_score(y, y_pred)

def get_label_strs(labels):
    label_str = []

    for l in labels:
        label_str.append(ID_TO_TRAFFIC[l])
    return label_str

def saveCM(y_true, y_pred, labels, output_path):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.argmax(1)
    l_strs = get_label_strs(labels)
    cm = confusion_matrix(labels[y_true], labels[y_pred], labels=labels)
    img_util.plot_confusion_matrix(cm, l_strs, True, result_folder=output_path)
    return cm