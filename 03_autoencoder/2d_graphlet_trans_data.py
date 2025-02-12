from pathlib import Path
from keras import losses
import numpy as np

from utils.trainer_ae import TrainerAE
import utils.io_util as io_util

# model
from utils.ae_model import AE
from models.ae import CAE_2d_2500

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Config:
    source_dir = 'data/split/2559/1d'
    trans_dir = 'data/split/2d_graphlet_fl/'

    ae_dir = 'train_data/2d_ae_fl/'
    cae_dir = 'train_data/2d_cae_fl/'

    model_fn_ae = 'fl_ae_server'
    model_fn_cae = 'fl_cae_server'

    model_batch_size = 5000
    model_filter_num = 16
    model_dropout = 0.15

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


def get_trainer(model_dir: str, model_fn: str, is_ae: bool):
    model_path = io_util.get_folder(Path(model_dir), 'models')
    model_path = io_util.get_folder(model_path, model_fn)

    model = None
    if is_ae:
        model = AE(48, 59)
    else:
        model = CAE_2d_2500(filter_num=Config.model_filter_num, dropout=Config.model_dropout)

    trainer = TrainerAE(
        folder=model_path, file_name=model_fn, model=model,
        X_train=None, X_val=None,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, load=True)
    if is_ae:
        trainer.model.build((None, 59))
    else:
        trainer.model.build((None, 50, 50, 1))
    return trainer

def saveNpz(datas, y, output_path, npz_name):
    fn = Path().joinpath(output_path, '{}.npz'.format(npz_name))
    np.savez_compressed(fn, x=np.array(datas), y=y)

def trans_save_data(source_path: Path, source_fn: str, ae_encoder, cae_encoder, output_path: Path, output_fn: str):
    x_graphlet, x_cnn, y, l = io_util.load_data_2d_graphlet(source_path.joinpath(source_fn), 42)
    # print('x_graphlet', x_graphlet.shape)
    # print('x_cnn', x_cnn.shape)
    x_graphlet = ae_encoder.predict(x_graphlet)
    x_cnn = cae_encoder.predict(x_cnn)
    x_cnn = np.reshape(x_cnn, (-1, 512))
    # print('x_graphlet', x_graphlet.shape)
    # print('x_cnn', x_cnn.shape)

    x = np.concatenate((x_graphlet, x_cnn), axis=1)
    print('x', x.shape)

    saveNpz(x, y, output_path, output_fn)


def trans_data():
    source_path = Path(Config.source_dir)
    trans_path = Path(Config.trans_dir)
    trans_path.mkdir(parents=True, exist_ok=True)

    ae_trainer = get_trainer(Config.ae_dir, Config.model_fn_ae, True)
    cae_trainer = get_trainer(Config.cae_dir, Config.model_fn_cae, False)

    trans_save_data(source_path, 'all_train.npz',
                    ae_trainer.model.encoder, cae_trainer.model.encoder,
                    trans_path, 'all_train')
    trans_save_data(source_path, 'all_val.npz',
                    ae_trainer.model.encoder, cae_trainer.model.encoder,
                    trans_path, 'all_val')
    trans_save_data(source_path, 'all_test.npz',
                    ae_trainer.model.encoder, cae_trainer.model.encoder,
                    trans_path, 'all_test')


def main():
    trans_data()


if __name__ == "__main__":
    main()