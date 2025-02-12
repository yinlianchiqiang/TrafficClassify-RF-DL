from pathlib import Path

import tensorflow as tf

import utils.utils as utils
from utils.trainer_new import Trainer
import utils.t_cnn_model as model_cnn
import utils.io_util as io_util
import utils.img_util as img_util
import utils.rf_util as rf_util

class Config:
    source_dir = 'data/split/2559/1d'
    output_dir = 'train_data/lab19'

    mode = 'test'  # [train/test/draw_acc_loss/draw_cm/draw_acc_loss_fl_server]

    model_file_name = 'fl_2d_cnn_server'
    model_batch_size = 5000
    model_input_shape = 50
    model_filter_num = 16
    model_dropout = 0.15
    model_is_dnn = True
    model_optimizer = 'adam'
    model_loss_fn = 'categorical_crossentropy'
    model_metrics = 'acc'

# =================================================
#  Train
# =================================================

# def nouse_changeImg():
#     source_path = Path(Config.source_dir)
#     output_path = Path(Config.output_dir)
#     model_path = io_util.get_folder(output_path, 'models')
#     dataset_val_fn = Path().joinpath(source_path, 'all_val.npz')
#     x_val, y_val_onehot, l = io_util.load_train_data_npz(dataset_val_fn, 42)
#     x_val_cnn = np.reshape(x_val[:,59:], (-1, 50, 50))
#     img = x_val_cnn[0]
#     png_file_path = Path().joinpath(model_path, 'RRR.png')

def train(config: Config):
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    dataset_train_fn = Path().joinpath(source_path, 'all_train.npz')
    dataset_val_fn = Path().joinpath(source_path, 'all_val.npz')
    model_path = io_util.get_folder(output_path, 'models')

    model = model_cnn.CNN_2d_graphlet(input_shape=50,
                             filter_num=config.model_filter_num,
                             dropout=config.model_dropout,
                             is_dnn=config.model_is_dnn)
    print(model.summary())

    x_train_graphlet, x_train_cnn, y_train_onehot, l = io_util.load_data_2d_graphlet(dataset_train_fn, 42)
    x_val_graphlet, x_val_cnn, y_val_onehot, l = io_util.load_data_2d_graphlet(dataset_val_fn, 42)
    
    print('Prepare Trainer...')
    trainer = Trainer(
        folder=model_path, file_name=config.model_file_name, model=model,
        X_train=[x_train_graphlet, x_train_cnn], y_train=y_train_onehot,
        X_val=[x_val_graphlet, x_val_cnn], y_val=y_val_onehot,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, metrics=Config.model_metrics)
    print('Trainer prepared.')
    # trainer.train(7)
    trainer.train(100)

    return trainer, model

# =================================================
#  Test
# =================================================


def test(config: Config):
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)

    print(source_path)
    print(output_path)
    
    dataset_test_fn = Path().joinpath(source_path, 'all_test.npz')
    model_path = io_util.get_folder(output_path, 'models_2')

    # get data
    x_test_graphlet, x_test_cnn, y_test_onehot, labels = io_util.load_data_2d_graphlet(dataset_test_fn, 42)

    model = model_cnn.CNN_2d_graphlet(input_shape=50,
                             filter_num=config.model_filter_num,
                             dropout=config.model_dropout,
                             is_dnn=config.model_is_dnn)

    trainer = Trainer(
        folder=model_path, file_name=config.model_file_name, model=model,
        X_train=None, y_train=None,
        X_val=[x_test_graphlet, x_test_cnn], y_val=y_test_onehot,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, metrics=Config.model_metrics, load=True)

    cm,y_pred = trainer.confusion_matrix(labels)

    rf_util.get_TP_TN(cm)
    y_t, y_p = rf_util.get_yPred_and_yTrue_without_one_hot(y_test_onehot, y_pred) 

    print("//---- F1/Precision/Recall ----//")
    rf_util.get_F1_score(y_t, y_p)
    rf_util.get_precision_score(y_t, y_p)
    rf_util.get_recall_score(y_t, y_p)

    return "testing without trainer", model


def draw_acc_loss(config: Config):
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    output_img_path = io_util.get_folder(output_path, config.model_file_name + '_img')
    params = utils.load(f'{model_path}/{config.model_file_name}.checkpoint.pth.params')

    img_util.drawLossF({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)
    img_util.drawAccF({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)

def draw_acc_loss_fl_server(config: Config):
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    output_img_path = io_util.get_folder(output_path, config.model_file_name + '_img')
    params = utils.load(f'{model_path}/{config.model_file_name}.checkpoint.pth.params')

    img_util.drawLossF_FL_server({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)
    img_util.drawAccF_FL_server({'total_epoch': params['total_epoch'], 'history': params['history']}, output_img_path, True)


def draw_cm(config: Config):
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')
    output_img_path = io_util.get_folder(output_path, config.model_file_name + '_img')

    cm = utils.load(f'{model_path}/{config.model_file_name}_cm.pickle')
    params = utils.load(f'{model_path}/{config.model_file_name}_cm.params')

    img_util.plot_confusion_matrix(cm, params["label_str"], True, result_folder=output_img_path)


def main():
    # load 參數
    config = Config()

    tf.keras.backend.clear_session()

    if config.mode == 'train':
        print('===== train =====')
        return train(config)
    elif config.mode == 'test':
        print('===== test =====')
        return test(config)
    elif config.mode == 'draw_acc_loss':
        print('===== draw_acc_loss =====')
        return draw_acc_loss(config)
    elif config.mode == 'draw_acc_loss_fl_server':
        print('===== draw_acc_loss_fl_server =====')
        return draw_acc_loss_fl_server(config)
    elif config.mode == 'draw_cm':
        print('===== draw_cm =====')
        return draw_cm(config)
    else:
        pass


if __name__ == '__main__':
    main()
