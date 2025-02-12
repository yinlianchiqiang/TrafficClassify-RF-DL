from keras import backend as K
from argparse import ArgumentParser
import os
import pickle as pk
import numpy as np
import csv

def write_row(fn, row_data, mode='a'):
    """寫 csv 檔"""
    with open(fn, mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def gen_todo_list(directory, check=None):
    """取得需要執行的檔案 (check 是指定檔名判斷式 call back)"""
    files = os.listdir(directory)
    todo_list = []
    for f in files:
        fullpath = os.path.join(directory, f)
        if os.path.isfile(fullpath):
            if check is not None:
                if check(f):
                    todo_list.append(fullpath)
            else:
                todo_list.append(fullpath)
    return todo_list


def load(filename):
    """讀檔"""
    with open(filename, 'rb') as f:
        data = pk.load(f)
    return data


def dump(data, filename):
    """寫檔"""
    with open(filename, 'wb') as f:
        pk.dump(data, f)


def get_args():
    """集中式版取得執行參數"""
    parser = ArgumentParser()
    #parser.add_argument("pos1", help="positional argument 1")
    parser.add_argument("-m", "--mode", help="trainer mode [train/test]", dest="mode", default="train")
    parser.add_argument("-n", "--name", help="model name [model]", dest="model_name", default="model")
    parser.add_argument("-tt", "--task_type", help="task type [app/class]", dest="task_type", default="app")
    parser.add_argument("-bs", "--batch-size", help="batch size [256]", dest="batch_size", default=256, type=int)
    parser.add_argument("-db", "--debug", help="debug [False]", dest="debug", action="store_true")
    parser.add_argument("-l", "--load", help="load [False]", dest="load", action="store_true")
    args = parser.parse_args()
    return args


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def check(filename):
    return not '_class' in filename


def load_data(config, is_test=False):
    assert config.idx in range(config.partition)
    # if config.debug:
    #     max_data_len = 10
    # else:
    #     max_data_len = 10000
    max_data_len = 10000
    directory = 'data'
    train_rate = 0.64
    val_rate = 0.16
    todo_list = gen_todo_list(directory, check=check)

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for counter, filename in enumerate(todo_list):
        (tmpX, tmpy) = load(filename)
        # if config.task_type == 'class':
        #     tmpy = load('.'.join(filename.split('.')[:-1]) + '_class.pickle')
        tmpX, tmpy = tmpX[:max_data_len], tmpy[:max_data_len] # 各檔最多選擇 max_data_len 筆
        assert(len(tmpX) == len(tmpy))
        tmpX = np.array(tmpX)

        # 集中式資料集劃分 => [ train             || val           || test ]
        # 聯合式資料集劃分 => [ train 1 | train 2 || val 1 | val 2 || test ]

        train_num = int(len(tmpX) * train_rate)
        val_num = int(len(tmpX) * val_rate)

        train_partition = int(train_num / config.partition)
        val_partition = int(val_num / config.partition)

        train_start = config.idx * train_partition
        train_end = (config.idx + 1) * train_partition
        val_start = train_num + config.idx * val_partition
        val_end = train_num + (config.idx + 1) * val_partition

        X_train.extend(tmpX[train_start: train_end])
        y_train.extend(tmpy[train_start: train_end])
        X_val.extend(tmpX[val_start: val_end])
        y_val.extend(tmpy[val_start: val_end])

        X_test.extend(tmpX[train_num + val_num:])
        y_test.extend(tmpy[train_num + val_num:])

        del tmpX, tmpy
        print("\rLoading... {}/{}".format(counter+1, len(todo_list)))
    print('\r{} Data loaded.'.format(len(todo_list)))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
