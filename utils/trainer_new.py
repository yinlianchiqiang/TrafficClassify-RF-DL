from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.utils import *
from utils.constants import ID_TO_TRAFFIC
#import nni
import tensorflow as tf
import time


class Trainer():
    def __init__(self, folder, file_name, model, X_train, y_train, X_val, y_val, batch_size, optimizer='adam', loss_fn='mse', metrics='acc', load=False):
        self.folder = folder
        self.file_name = file_name
        self.model = model
        # data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        # trainer attributes
        self.total_epoch = 0
        self.history = []
        self.batch_size = batch_size
        # train param
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.monitor = 'val_{}'.format(metrics)
        self.metrics_fn = 'acc'
        self.monitor_mode = 'max'
        self._set_param(metrics)

        self.cm = None 
        self.label_str = []

        if load:
            self.load()

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=[self.metrics_fn])

        # self.best_f1 = 0

    def _set_param(self, metrics):
        if metrics == 'f1_score':
            self.metrics_fn = f1_score
            self.monitor_mode = 'max'
        elif metrics == 'acc':
            self.metrics_fn = 'acc'
            self.monitor_mode = 'max'
        elif metrics == 'mse':
            self.metrics_fn = 'mse'
            self.monitor = 'val_mean_squared_error'
            self.monitor_mode = 'min'

    def train(self, epoch=1, verbose=True):
        self.total_epoch += epoch
        # 集中式版使用訓練 (fit + 存紀錄/參數 + evaluate)
        #filepath = f'{self.folder}/{self.file_name}.checkpoint.pth.h5'
        #print('a')
        filepath = f'{self.folder}/{self.file_name}.checkpoint.pth.h5'
        checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1,
                                     save_best_only=True, save_weights_only=True, mode=self.monitor_mode)
        #print('b')
        # best_checkpoint = ModelCheckpoint(filepath, monitor='f1', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]  # , best_checkpoint]
        #NNI callback
        '''callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
        )'''
        #print('c')
        hist = self.model.fit(self.X_train, self.y_train,
                              validation_data=(self.X_val, self.y_val),
                              epochs=epoch,
                              batch_size=self.batch_size,
                              verbose=verbose,
                              callbacks=callbacks_list) #callbacks_list)
        print('d')

        score = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        print('e')
        print('Val loss:', score[0])
        print('Val accuracy:', score[1])
        #nni.report_final_result(score[1])
        self.history.append(hist.history)
        self.save()

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, accuracy

    def evaluate_fl_server(self):
        # FL server 用 evaluate
        self.total_epoch += 1
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        self.history.append({'test_acc': accuracy,
                            'test_loss': loss})
        filepath = f'{self.folder}/{self.file_name}.checkpoint.pth.h5'
        self.model.save_weights(filepath)
        self.save()
        return loss, accuracy

    def fit(self, epoch=1, verbose=True):
        self.total_epoch += epoch
        # FL 版使用訓練  (fit + 存紀錄/參數)
        filepath = f'{self.folder}/{self.file_name}.checkpoint.pth.h5'
        checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1,
                                     save_best_only=True, save_weights_only=True, mode=self.monitor_mode)
        callbacks_list = [checkpoint]
        hist = self.model.fit(self.X_train, self.y_train,
                              validation_data=(self.X_val, self.y_val),
                              epochs=epoch,
                              batch_size=self.batch_size,
                              verbose=verbose,
                              callbacks=callbacks_list)
        self.history.append(hist.history)
        self.save()

    # ====================================================
    # Trainer save/load
    # ====================================================

    def save(self):
        params = {'total_epoch': self.total_epoch,
                  'batch_size': self.batch_size,
                  'history': self.history}
        dump(params, f'{self.folder}/{self.file_name}.checkpoint.pth.params')

    def load(self):
        params = load(f'{self.folder}/{self.file_name}.checkpoint.pth.params')
        self.total_epoch = params['total_epoch']
        self.batch_size = params['batch_size']
        self.history = params['history']
        self.model.load_weights(f'{self.folder}/{self.file_name}.checkpoint.pth.h5')
        print("load success!!")

    # ====================================================
    # Trainer confusion_matrix
    # ====================================================

    def confusion_matrix(self, labels):
        label_str = []

        for l in labels:
            label_str.append(ID_TO_TRAFFIC[l])

        loss, accuracy = self.evaluate()
        print('Test Loss:', loss)
        print('Test Accuracy:', accuracy)

        start_time = time.time() 

        y_pred = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=True)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"\n!!Execution time: {elapsed_time:.6f} seconds\n")

        re_y_pred = y_pred
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(labels[self.y_val.argmax(1)], labels[y_pred], labels=labels)

        self.label_str = label_str
        self.cm = cm
        dump({"label_str": label_str}, f'{self.folder}/{self.file_name}_cm.params')
        dump(cm, f'{self.folder}/{self.file_name}_cm.pickle')

        return cm,re_y_pred