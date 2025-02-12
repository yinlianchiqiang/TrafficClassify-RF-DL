from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import losses

import utils.utils as utils
import utils.io_util as io_util
import utils.img_util as img_util
from utils.ae_model import AE
from utils.trainer_ae import TrainerAE

class Config:
    source_dir = 'data/split/single_1d_1500'
    output_dir = 'train_data/1500_port_ae_fl_5000/'
    output_trans_dir = 'data/split/1500_port_ae_fl_5000/1'

    mode = 'trans_data'  # [train / draw_loss / trans_data / ae_img ]

    model_file_name = 'fl_ae_server'
    model_batch_size = 5000

    model_input_shape = 1500
    model_encoded_output = 256

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


def saveNpz(datas, y, output_path, npz_name):
    fn = Path().joinpath(output_path, '{}.npz'.format(npz_name))
    np.savez_compressed(fn, x=np.array(datas), y=y)


# 顯示效果
def getAEimg(ori_data, ae, fn, img_o_w, img_o_h, img_e_w, img_e_h, num=10):
    plt.figure(figsize=(20, 4))
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = ae.encoder.predict(ori_data[:num])
    decoded_imgs = ae.decoder.predict(encoded_imgs)
    for i in range(num):
        # display original
        ax = plt.subplot(3, num, i + 1)
        tmp_ori = ori_data[i]
        tmp_decoded = decoded_imgs[i]
        if Config.model_input_shape == 59:
            pad_w = img_o_h * img_o_w - Config.model_input_shape
            tmp_ori = np.pad(tmp_ori, pad_width=(0, pad_w), constant_values=255)
            tmp_decoded = np.pad(tmp_decoded, pad_width=(0, pad_w), constant_values=255)

        plt.imshow(tmp_ori.reshape(img_o_h, img_o_w))
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # encoded_imgs
        ax = plt.subplot(3, num, i + 1 + num)
        plt.imshow(encoded_imgs[i].reshape(img_e_h, img_e_w))
        plt.title("encoded")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, num, i + 1 + num + num)
        plt.imshow(tmp_decoded.reshape(img_o_h, img_o_w))
        plt.title("decoded")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(fn)
    plt.close()


def trans_save_data(source_path: Path, source_fn: str, encoder, output_path: Path, output_fn: str):
    x, y = io_util.load_data_ae(source_path.joinpath(source_fn), 42)
    x_ae = encoder.predict(x)
    saveNpz(x_ae, y, output_path, output_fn)


def train():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    model_path.mkdir(parents=True, exist_ok=True)

    # 訓練參數
    train_epochs = 100

    # loaddata
    x_train, y_train = io_util.load_data_ae(source_path.joinpath('all_train.npz'), 42)
    x_val, y_val = io_util.load_data_ae(source_path.joinpath('all_val.npz'), 42)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    # AE train
    ae = AE(Config.model_encoded_output, Config.model_input_shape)

    print('Prepare Trainer...')
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=ae,
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
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)

    # model
    ae = AE(Config.model_encoded_output, Config.model_input_shape)
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=ae,
        X_train=None, X_val=None,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, load=True)
    trainer.model.build((None, Config.model_input_shape))

    trans_save_data(source_path, 'all_train.npz', trainer.model.encoder, output_trans_path, 'all_train')
    trans_save_data(source_path, 'all_val.npz', trainer.model.encoder, output_trans_path, 'all_val')
    trans_save_data(source_path, 'all_test.npz', trainer.model.encoder, output_trans_path, 'all_test')


def ae_img():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    output_img_path = io_util.get_folder(output_path, Config.model_file_name + '_img')

    # 參數
    img_o_w = 10
    img_o_h = 6
    img_e_w = 8
    img_e_h = 6

    # load data
    x_val, y_val = io_util.load_data_ae(source_path.joinpath('all_val.npz'), 42)

    # model
    ae = AE(Config.model_encoded_output, Config.model_input_shape)
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=ae,
        X_train=None, X_val=None,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, load=True)

    trainer.model.build((None, Config.model_input_shape))

    getAEimg(x_val, trainer.model, Path().joinpath(output_img_path, 'auto_ae.png'), img_o_w, img_o_h, img_e_w, img_e_h, 10)


def main():
    tf.keras.backend.clear_session()
    if Config.mode == 'train':
        print('===== train =====')
        return train()
    elif Config.mode == 'draw_loss':
        print('===== draw_loss =====')
        return draw_loss()
    elif Config.mode == 'trans_data':
        print('===== trans_data =====')
        return trans_data()
    elif Config.mode == 'ae_img':
        print('===== ae_img =====')
        return ae_img()
    else:
        pass


if __name__ == "__main__":
    main()
