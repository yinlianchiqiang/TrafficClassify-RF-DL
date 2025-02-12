import csv
from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer


def write_row(fn, row_data, mode='a'):
    """寫 csv 檔"""
    with open(fn, mode, newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def load_train_data_npz(file_path, random_state, dims=0):
    data = np.load(file_path, allow_pickle=True)

    x, y = data['x'], data['y']

    if dims > 0:
        # normalize X
        x = np.expand_dims(x, dims)
    #x = np.array(x,dtype = np.float16)
    x, y = shuffle(x, y, random_state=random_state)

    label_encoder = LabelBinarizer()
    y_onehot = label_encoder.fit_transform(y)

    return x, y_onehot, label_encoder.classes_


def load_data_ae(file_path, random_state):
    data = np.load(file_path, allow_pickle=True)

    x, y = data['x'], data['y']
    x, y = shuffle(x, y, random_state=random_state)

    return x, y


def load_data_2d_graphlet(file_path, random_state):
    x, y, l = load_train_data_npz(file_path, random_state)

    x_graphlet = x[:,:59]
    x_cnn = np.reshape(x[:,59:], (-1, 50, 50))
    x_cnn = np.expand_dims(x_cnn, 3)
    
    return x_graphlet, x_cnn, y, l


def get_folder(root_path, folder):
    path = Path().joinpath(root_path, folder)
    path.mkdir(parents=True, exist_ok=True)
    return path
