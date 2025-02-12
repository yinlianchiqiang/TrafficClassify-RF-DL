from keras.callbacks import ModelCheckpoint
from utils.utils import *


class TrainerAE():
    def __init__(self, folder, file_name, model, X_train, X_val, batch_size, optimizer='adam', loss_fn='mse', load=False):
        self.folder = folder
        self.file_name = file_name
        self.model = model
        # data
        self.X_train = X_train
        self.X_val = X_val
        # trainer attributes
        self.total_epoch = 0
        self.history = []
        self.batch_size = batch_size
        # train param
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.monitor = 'val_loss'
        self.monitor_mode = 'min'

        if load:
            self.load()

        self.model.compile(optimizer=optimizer, loss=loss_fn)

    def train(self, epoch=1, verbose=True):
        self.total_epoch += epoch
        # 集中式版使用訓練 (fit + 存紀錄/參數 + evaluate)
        filepath = f'{self.folder}/{self.file_name}.checkpoint'
        checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1,
                                     save_best_only=True, save_weights_only=True, mode=self.monitor_mode)
        callbacks_list = [checkpoint]

        hist = self.model.fit(self.X_train, self.X_train,  # Label 也設為 x_train
                              validation_data=(self.X_val, self.X_val),
                              epochs=epoch,
                              batch_size=self.batch_size,
                              shuffle=True,
                              verbose=verbose,
                              callbacks=callbacks_list)

        score = self.evaluate()
        print('Val loss:', score)
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
        self.model.load_weights(f'{self.folder}/{self.file_name}.checkpoint').expect_partial()

        print("load success!!")

    def evaluate(self):
        loss = self.model.evaluate(self.X_val, self.X_val, verbose=0)
        return loss

    def evaluate_fl_server(self):
        # FL server 用 evaluate
        self.total_epoch += 1
        loss = self.evaluate()
        self.history.append({'test_loss': loss})
        filepath = f'{self.folder}/{self.file_name}.checkpoint'
        self.model.save_weights(filepath)
        self.save()
        return loss

    def fit(self, epoch=1, verbose=True):
        self.total_epoch += epoch
        # FL 版使用訓練  (fit + 存紀錄/參數)
        filepath = f'{self.folder}/{self.file_name}.checkpoint'
        checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1,
                                     save_best_only=True, save_weights_only=True, mode=self.monitor_mode)
        callbacks_list = [checkpoint]
        hist = self.model.fit(self.X_train, self.X_train,
                              validation_data=(self.X_val, self.X_val),
                              epochs=epoch,
                              batch_size=self.batch_size,
                              verbose=verbose,
                              callbacks=callbacks_list)
        self.history.append(hist.history)
        self.save()
