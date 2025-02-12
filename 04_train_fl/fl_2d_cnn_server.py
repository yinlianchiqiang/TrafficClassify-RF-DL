
from pathlib import Path
import flwr as fl
import tensorflow as tf

import utils.t_cnn_model as model_cnn
from utils.trainer_new import Trainer
import utils.io_util as io_util


class Config:
    source_dir = 'data/split/2559/1d'
    output_dir = 'train_data/lab19'
    
    model_file_name = 'fl_2d_cnn_server'
    model_batch_size = 1000
    model_input_shape = 50
    model_filter_num = 16
    model_dropout = 0.05
    model_is_dnn = True
    model_optimizer = 'adam'
    model_loss_fn = 'categorical_crossentropy'
    model_metrics = 'acc'


def get_eval_fn(trainer):
    """自訂驗證方法 (紀錄loss, accuracy)。Return an evaluation function"""

    def evaluate(parameters):
        trainer.model.set_weights(parameters)
        loss, accuracy = trainer.evaluate_fl_server()
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    tf.keras.backend.clear_session()

    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    dataset_test_fn = Path().joinpath(source_path, 'all_test.npz')
    model_path = io_util.get_folder(output_path, 'models')
    
    # 取得資料
    
    x_test_graphlet, x_test_cnn, y_test_onehot, l = io_util.load_data_2d_graphlet(dataset_test_fn, 42)
    print(x_test_graphlet.shape, x_test_cnn.shape, y_test_onehot.shape)

    # train
    model = model_cnn.CNN_2d_graphlet(input_shape=Config.model_input_shape,
                             filter_num=Config.model_filter_num,
                             dropout=Config.model_dropout,
                             is_dnn=Config.model_is_dnn)
    trainer = Trainer(
        folder=model_path, file_name=Config.model_file_name, model=model,
        X_train=None, y_train=None,
        X_val=[x_test_graphlet, x_test_cnn], y_val=y_test_onehot,
        batch_size=Config.model_batch_size, optimizer=Config.model_optimizer,
        loss_fn=Config.model_loss_fn, metrics=Config.model_metrics, load=False)

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

    trainer.confusion_matrix(l)


if __name__ == "__main__":
    main()
