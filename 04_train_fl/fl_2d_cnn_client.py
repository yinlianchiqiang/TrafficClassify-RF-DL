import argparse
import os
from pathlib import Path

import flwr as fl

import utils.t_cnn_model as model_cnn
from utils.trainer_new import Trainer
import utils.io_util as io_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Config:
    source_dir = 'data/split/2559/1d'
    output_dir = 'train_data/lab19'

    model_file_name = 'fl_2d_cnn_dnn_client'
    model_batch_size = 1000
    model_input_shape = 50
    model_filter_num = 16
    model_dropout = 0.05
    model_is_dnn = True
    model_optimizer = 'adam'
    model_loss_fn = 'categorical_crossentropy'
    model_metrics = 'acc'


# 實作 Flower client 的 Abstract class
class TestClient(fl.client.NumPyClient):
    def __init__(self, model_folder, model, x_train, y_train, x_val, y_val, idx):
        self.trainer = Trainer(
            folder=model_folder, file_name=f'{Config.model_file_name}_{str(idx)}', model=model,
            X_train=x_train, y_train=y_train,
            X_val=x_val, y_val=y_val,
            batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
            loss_fn=Config.model_loss_fn, metrics=Config.model_metrics, load=False)

    def get_parameters(self):
        return self.trainer.model.get_weights()

    def fit(self, parameters, config):
        self.trainer.model.set_weights(parameters)
        self.trainer.fit()
        return self.trainer.model.get_weights(), len(self.trainer.X_train), {}

    def evaluate(self, parameters, config):
        self.trainer.model.set_weights(parameters)
        loss, accuracy = self.trainer.evaluate()
        return loss, len(self.trainer.X_val), {"accuracy": accuracy}


def main():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')

    # get argument
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-i", "--idx", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    print(args)

    # data
    
    dataset_train_fn = Path().joinpath(source_path, f'c{str(args.idx)}_train.npz')
    dataset_test_fn = Path().joinpath(source_path, f'c{str(args.idx)}_val.npz')
    x_train_graphlet, x_train_cnn, y_train, l = io_util.load_data_2d_graphlet(dataset_train_fn, 42)
    x_test_graphlet, x_test_cnn, y_test, l = io_util.load_data_2d_graphlet(dataset_test_fn, 42)

    # train
    model = model_cnn.CNN_2d_graphlet(input_shape=Config.model_input_shape,
                             filter_num=Config.model_filter_num,
                             dropout=Config.model_dropout,
                             is_dnn=Config.model_is_dnn)

    # 啟動 client
    _client = TestClient(model_path, model, [x_train_graphlet, x_train_cnn], y_train, [x_test_graphlet, x_test_cnn], y_test, args.idx)
    fl.client.start_numpy_client("localhost:4326", client=_client)


if __name__ == "__main__":
    main()
