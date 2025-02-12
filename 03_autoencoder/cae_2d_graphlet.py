from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import losses

import utils.utils as utils
import utils.io_util as io_util
import utils.img_util as img_util

from models.ae import CAE_2d_2500
from utils.trainer_ae import TrainerAE


class Config:
    source_dir = 'data/split/2559/1d'
    output_dir = 'train_data/2d_cae_fl/'

    mode = 'draw_loss_fl_server'  # [train/draw_loss/draw_loss_fl_server]
    data_type = 2559  # [1500/2559]

    model_file_name = 'fl_cae_server'
    model_batch_size = 5000

    model_filter_num = 16
    model_dropout = 0.15

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


def saveNpz(datas, y, output_path, npz_name):
    fn = Path().joinpath(output_path, '{}.npz'.format(npz_name))
    np.savez_compressed(fn, x=np.array(datas), y=y)


def train():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    model_path.mkdir(parents=True, exist_ok=True)

    # AE train
    model = CAE_2d_2500(filter_num=Config.model_filter_num, dropout=Config.model_dropout)
    # print(model.build((None, 50, 50, 1)))
    # print(model.encoder.summary())
    # print(model.decoder.summary())

    # 訓練參數
    train_epochs = 100

    # loaddata
    dataset_train_fn = source_path.joinpath('all_train.npz')
    dataset_val_fn = source_path.joinpath('all_val.npz')
    tmp, x_train_cnn, tmp, tmp = io_util.load_data_2d_graphlet(dataset_train_fn, 42)
    tmp, x_val_cnn, tmp, tmp = io_util.load_data_2d_graphlet(dataset_val_fn, 42)

    del tmp

    print(x_train_cnn.shape)
    print(x_val_cnn.shape)

    print('Prepare Trainer...')
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=model,
        X_train=x_train_cnn, X_val=x_val_cnn,
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

def draw_loss_fl_server(config: Config):
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    output_img_path = io_util.get_folder(output_path, config.model_file_name + '_img')
    params = utils.load(f'{model_path}/{config.model_file_name}/{config.model_file_name}.checkpoint.pth.params')

    img_util.drawLossF_FL_server({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)


def main():
    config = Config()
    tf.keras.backend.clear_session()

    if Config.mode == 'train':
        print('===== train =====')
        return train()
    elif Config.mode == 'draw_loss':
        print('===== draw_loss =====')
        return draw_loss()
    elif Config.mode == 'draw_loss_fl_server':
        print('===== draw_loss_fl_server =====')
        return draw_loss_fl_server(config)
    else:
        pass


if __name__ == "__main__":
    main()
