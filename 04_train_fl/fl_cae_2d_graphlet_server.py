
from pathlib import Path
import flwr as fl
import tensorflow as tf
from keras import losses

from tensorflow.keras import mixed_precision

from models.ae import CAE_2d_2500
from utils.trainer_ae import TrainerAE
import utils.io_util as io_util
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Config:
    source_dir = 'data/split/2559/1d'
    output_dir = 'train_data/2d_cae_fl'

    data_type = 2559  # [1500/2559]

    model_file_name = 'fl_cae_server'
    model_batch_size = 1000

    model_filter_num = 16
    model_dropout = 0.15

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


def get_eval_fn(trainer: TrainerAE):
    """自訂驗證方法 (紀錄loss, accuracy)。Return an evaluation function"""

    def evaluate(parameters):
        trainer.model.set_weights(parameters)
        loss = trainer.evaluate_fl_server()
        return loss, {"loss": loss}

    return evaluate


def main():
    tf.keras.backend.clear_session()

    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    dataset_test_fn = Path().joinpath(source_path, 'all_test.npz')
    model_path = io_util.get_folder(output_path, 'models')
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    model_path.mkdir(parents=True, exist_ok=True)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # 取得資料
    tmp, x_test, tmp, tmp= io_util.load_data_2d_graphlet(dataset_test_fn, 42)
    del tmp

    # train
    model = CAE_2d_2500(filter_num=Config.model_filter_num, dropout=Config.model_dropout)
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=model,
        X_train=x_test, X_val=x_test,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, load=False)
    trainer.model.build((None, 50, 50, 1))
    # trainer.model.summary()

    # FedAvg 給參數
    _strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=get_eval_fn(trainer),
    )

    # 啟動 server
    fl.server.start_server(
        "localhost:4326",
        force_final_distributed_eval=True,
        strategy=_strategy,
        config={
            "num_rounds": 100
        }
    )


if __name__ == "__main__":
    main()
