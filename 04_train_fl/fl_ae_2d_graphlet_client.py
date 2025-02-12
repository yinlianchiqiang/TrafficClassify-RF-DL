import argparse
import os
from pathlib import Path

import flwr as fl
from keras import losses

from utils.ae_model import AE
from utils.trainer_ae import TrainerAE
import utils.io_util as io_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Config:
    source_dir = 'data/split/2559/1d'
    output_dir = 'train_data/2d_ae_fl'

    model_file_name = 'fl_ae_client'
    model_batch_size = 5000

    model_input_shape = 59
    model_encoded_output = 48

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


# 實作 Flower client 的 Abstract class
class TestClient(fl.client.NumPyClient):
    def __init__(self, model_folder, model, x_train, x_val, idx):
        self.trainer = TrainerAE(
            folder=model_folder, file_name=f'{Config.model_file_name}_{str(idx)}', model=model,
            X_train=x_train, X_val=x_val,
            batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
            loss_fn=Config.model_loss_fn)
        self.trainer.model.build((None, Config.model_input_shape))

    def get_parameters(self):
        return self.trainer.model.get_weights()

    def fit(self, parameters, config):
        self.trainer.model.set_weights(parameters)
        self.trainer.fit()
        return self.trainer.model.get_weights(), len(self.trainer.X_train), {}

    def evaluate(self, parameters, config):
        self.trainer.model.set_weights(parameters)
        loss = self.trainer.evaluate()
        return loss, len(self.trainer.X_val), {"loss": loss}


def main():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    model_path = io_util.get_folder(output_path, 'models')

    # get argument
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("-i", "--idx", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    print(args)

    model_path = io_util.get_folder(model_path, f'{Config.model_file_name}_{str(args.idx)}')

    # data
    dataset_train_fn = Path().joinpath(source_path, f'c{str(args.idx)}_train.npz')
    dataset_test_fn = Path().joinpath(source_path, f'c{str(args.idx)}_val.npz')
    x_train, tmp, tmp, tmp= io_util.load_data_2d_graphlet(dataset_train_fn, 42)
    x_test, tmp, tmp, tmp = io_util.load_data_2d_graphlet(dataset_test_fn, 42)
    del tmp

    # train
    ae = AE(Config.model_encoded_output, Config.model_input_shape)

    # 啟動 client
    _client = TestClient(model_path, ae, x_train, x_test, args.idx)
    fl.client.start_numpy_client("localhost:4326", client=_client)


if __name__ == "__main__":
    main()
