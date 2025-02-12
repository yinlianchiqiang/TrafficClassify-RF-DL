from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import losses

import utils.utils as utils
import utils.io_util as io_util
import utils.img_util as img_util

from models.ae import CAE_1d_1500
from utils.ae_model import AE
from utils.trainer_ae import TrainerAE
import gc
from numba import cuda


class Config:
    source_dir = 'data/split/single_1d_1500'
    output_dir = 'train_data/1500_cae_fl_5000/'
    output_trans_dir = 'data/split/1500_cae_fl_5000/'

    mode = 'trans_data'  # [train/draw_loss/trans_data/draw_loss_fl_server]
    data_type = 1500  # [1500/2559]

    model_file_name = 'fl_cae_server'
    model_batch_size = 5000

    model_filter_num = 16
    model_dropout = 0.15

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


def saveNpz(datas, y, output_path, npz_name):
    fn = Path().joinpath(output_path, '{}.npz'.format(npz_name))
    np.savez_compressed(fn, x=np.array(datas), y=y)


def trans_save_data(source_path: Path, source_fn: str, encoder, output_path: Path, output_fn: str):
    x, y, l = io_util.load_train_data_npz(source_path.joinpath(source_fn), 42, 2)
    x_ae = encoder.predict(x)
    saveNpz(x_ae, y, output_path, output_fn)


def train():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    model_path.mkdir(parents=True, exist_ok=True)

    # AE train
    model = CAE_1d_1500(filter_num=Config.model_filter_num, dropout=Config.model_dropout)
    # print(model.build((None, 1500, 1)))
    # print(model.encoder.summary())
    # print(model.decoder.summary())

    # 訓練參數
    train_epochs = 100

    # loaddata
    x_train, y_train, l = io_util.load_train_data_npz(source_path.joinpath('all_train.npz'), 42, 2)
    x_val, y_val, l = io_util.load_train_data_npz(source_path.joinpath('all_val.npz'), 42, 2)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    print('Prepare Trainer...')
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=model,
        X_train=x_train, X_val=x_val,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn)
    print('Trainer prepared.')

    trainer.train(train_epochs)

    print(trainer.total_epoch)


def draw_loss():
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    output_img_path = io_util.get_folder(output_path, Config.model_file_name + '_img')
    params = utils.load(f'{model_path}/{Config.model_file_name}.checkpoint.pth.params')

    img_util.drawLossF({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)


def trans_data():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    output_trans_path = Path(Config.output_trans_dir)
    output_trans_path.mkdir(parents=True, exist_ok=True)
    model_path = io_util.get_folder(output_path, 'models')#models
    model_path = io_util.get_folder(model_path, Config.model_file_name)

    # model
    model = CAE_1d_1500(filter_num=Config.model_filter_num, dropout=Config.model_dropout)
    model.build((None, 1500, 1))
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=model,
        X_train=None, X_val=None,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, load=True)

    trans_save_data(source_path, 'all_train.npz', trainer.model.encoder, output_trans_path, 'all_train')
    trans_save_data(source_path, 'all_val.npz', trainer.model.encoder, output_trans_path, 'all_val')
    trans_save_data(source_path, 'all_test.npz', trainer.model.encoder, output_trans_path, 'all_test')

def draw_loss_fl_server(config: Config):
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    output_img_path = io_util.get_folder(output_path, config.model_file_name + '_img')
    params = utils.load(f'{model_path}/{config.model_file_name}/{config.model_file_name}.checkpoint.pth.params')

    img_util.drawLossF_FL_server({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)


def main():
    config = Config()
    gc.collect()
    tf.keras.backend.clear_session()
    cuda.close()
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    #    except RuntimeError as e:
    #        print(e)

    if Config.mode == 'train':
        print('===== train =====')
        return train()
    elif Config.mode == 'draw_loss':
        print('===== draw_loss =====')
        return draw_loss()
    elif Config.mode == 'trans_data':
        print('===== trans_data =====')
        return trans_data()
    elif Config.mode == 'draw_loss_fl_server':
        print('===== draw_loss_fl_server =====')
        return draw_loss_fl_server(config)
    else:
        pass


if __name__ == "__main__":
    main()
