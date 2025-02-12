from argparse import ArgumentParser

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import numpy as np

import utils as utils
from utils.trainer import Trainer
import utils.t_cnn_model as model_cnn

data_dir = 'split/2d_1024_payload'

def get_args():
    """集中式版取得執行參數"""
    parser = ArgumentParser()
    #parser.add_argument("pos1", help="positional argument 1")
    parser.add_argument("-m", "--mode", help="trainer mode [train/test]", dest="mode", default="train")
    parser.add_argument("-n", "--name", help="model name [model]", dest="model_name", default="model")
    parser.add_argument("-bs", "--batch-size", help="batch size [256]", dest="batch_size", default=256, type=int)
    parser.add_argument("-l", "--load", help="load [False]", dest="load", action="store_true")
    args = parser.parse_args()
    return args

def loaddata(file_path, random_state):
    data = np.load(file_path, allow_pickle=True)

    x, y = data['x'], data['y']
    # normalize X
    x = np.expand_dims(x, 3)
    x, y = shuffle(x, y, random_state=random_state)

    return x, y

# =================================================
#  Train
# =================================================


def train(config):
    print('===== train =====')

    model = model_cnn.CNN_2d(input_shape=32,
                             filter_num=16,
                             dropout=0.05,
                             is_dnn=True)
    print(model.summary())

    x_train, y_train = loaddata('{}/all_train.npz'.format(data_dir), 42)
    x_val, y_val = loaddata('{}/all_val.npz'.format(data_dir), 22)

    # 把 y 的 string 做成 one hot encoding 形式
    label_encoder = LabelBinarizer()
    y_train_onehot = label_encoder.fit_transform(y_train)
    y_val_onehot = label_encoder.transform(y_val)

    print('Prepare Trainer...')
    trainer = Trainer(config, model, x_train, y_train_onehot, x_val, y_val_onehot, loss_fn='categorical_crossentropy', metrics='acc')
    print('Trainer prepared.')
    # trainer.train(7)
    trainer.train(100)
    return trainer, model

# =================================================
#  Test
# =================================================


def test(config):
    print('===== test =====')

    model = model_cnn.CNN_2d(input_shape=32,
                             filter_num=16,
                             dropout=0.05,
                             is_dnn=True)

    x_test, y_test = loaddata('{}/all_test.npz'.format(data_dir), 66)

    label_encoder = LabelBinarizer()
    y_test_onehot = label_encoder.fit_transform(y_test)

    model.load_weights('train_data/models/{}.checkpoint.pth.h5'.format(config.model_name))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.evaluate(x_test, y_test_onehot, batch_size=config.batch_size, verbose=True))
    y_pred = model.predict(x_test, batch_size=config.batch_size, verbose=True)
    y_pred = np.argmax(y_pred, axis=1)
    labels = label_encoder.classes_
    print('=====================')
    print(labels)
    print('=====================')
    cm = confusion_matrix(labels[y_test_onehot.argmax(1)], labels[y_pred], labels=labels)
    utils.dump(cm, 'train_data/objs/cm_{}.pickle'.format(config.model_name))
    return "testing without trainer", model


def main():
    # load 參數
    config = get_args()
    config.name = config.model_name
    config.load = False

    if config.mode == 'train':
        print('===== train =====')
        return train(config)
    elif config.mode == 'test':
        print('===== test =====')
        return test(config)
    else:
        pass


if __name__ == '__main__':
    main()
