
from pathlib import Path
import flwr as fl
import tensorflow as tf
from keras import losses
from tensorflow.keras import mixed_precision
from utils.ae_model import AE, SAE
from utils.trainer_ae import TrainerAE
import utils.io_util as io_util


class Config:
    source_dir = 'data/split/single_1d_1500'
    output_dir = 'train_data/1500_port_ae_fl_5000'

    model_file_name = 'fl_ae_server'
    model_batch_size = 2000

    model_input_shape = 1500
    model_encoded_output = 256

    model_optimizer = 'adam'
    model_loss_fn = losses.MeanSquaredError()  # binary_crossentropy


def get_eval_fn(trainer: TrainerAE):
    """自訂驗證方法 (紀錄loss, accuracy)。Return an evaluation function"""

    def evaluate(parameters):
        trainer.model.set_weights(parameters)
        trainer.model.build((None, Config.model_input_shape))
        loss = trainer.evaluate_fl_server()
        return loss, {"loss": loss}

    return evaluate


def main():
    tf.keras.backend.clear_session()

    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    dataset_test_fn = Path().joinpath(source_path, 'all_test.npz')
    model_path = io_util.get_folder(output_path, 'models')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    model_path = io_util.get_folder(model_path, Config.model_file_name)
    model_path.mkdir(parents=True, exist_ok=True)

    # 取得資料
    x_test, y_test = io_util.load_data_ae(dataset_test_fn, 42)

    # train
    ae = AE(Config.model_encoded_output, Config.model_input_shape)
    trainer = TrainerAE(
        folder=model_path, file_name=Config.model_file_name, model=ae,
        X_train=x_test, X_val=x_test,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn)
    trainer.model.build((None, Config.model_input_shape))
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
