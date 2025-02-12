from pathlib import Path
import tensorflow as tf
import utils.utils as utils
from utils.trainer_new import Trainer
import utils.t_cnn_model as model_cnn
import utils.io_util as io_util
import utils.img_util as img_util
#import nni


params = {
  'model_batch_size': 128,
  'model_filter_num': 16,
  'model_dropout' : 0.2,
}
#optimized_params = nni.get_next_parameter()
#params.update(optimized_params)
#print(params)

class Config:
    source_dir = 'data/split/single_1d_1024/exp1_1'
    output_dir = 'train_data/single_1d_1024/exp1_1'

    mode = 'train'  # [train/test/draw_acc_loss/draw_cm/draw_acc_loss_fl_server]

    model_file_name = 'cl_cnn'
    model_batch_size = params['model_batch_size']#125 #
    model_input_shape = 1024
    model_filter_num = params['model_filter_num']#16 #
    model_dropout = params['model_dropout']#0.05 #0.2-0.5
    model_is_dnn = True
    model_optimizer = 'adam'
    model_loss_fn = 'categorical_crossentropy'
    model_metrics = 'acc'

# =================================================
#  Train
# =================================================


def train(config: Config):
    #print('1')
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    dataset_train_fn = Path().joinpath(source_path, 'all_train.npz')
    dataset_val_fn = Path().joinpath(source_path, 'all_val.npz')
    model_path = io_util.get_folder(output_path, 'models')

    model = model_cnn.CNN_1d(input_shape=config.model_input_shape,
                             filter_num=config.model_filter_num,
                             dropout=config.model_dropout,
                             is_dnn=config.model_is_dnn)
    print(model.summary())
    #print('2')
    x_train, y_train_onehot, l = io_util.load_train_data_npz(dataset_train_fn, 42, 2)
    x_val, y_val_onehot, l = io_util.load_train_data_npz(dataset_val_fn, 42, 2)

    print('Prepare Trainer...')
    trainer = Trainer(
        folder=model_path, file_name=config.model_file_name, model=model,
        X_train=x_train, y_train=y_train_onehot,
        X_val=x_val, y_val=y_val_onehot,
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
    dataset_test_fn = Path().joinpath(source_path, 'all_test.npz')
    model_path = io_util.get_folder(output_path, 'models')

    # get data
    x_test, y_test_onehot, labels = io_util.load_train_data_npz(dataset_test_fn, 42, 2)

    model = model_cnn.CNN_1d(input_shape=config.model_input_shape,
                             filter_num=config.model_filter_num,
                             dropout=config.model_dropout,
                             is_dnn=config.model_is_dnn)

    trainer = Trainer(
        folder=model_path, file_name=config.model_file_name, model=model,
        X_train=None, y_train=None,
        X_val=x_test, y_val=y_test_onehot,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, metrics=Config.model_metrics, load=True)

    trainer.confusion_matrix(labels)

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
   
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf.keras.backend.clear_session()
    #gpus = tf.config.list_physical_devices('GPU')
    '''if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)'''

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
