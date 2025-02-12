from pathlib import Path

import utils.io_util as io_util
import utils.rf_util as rf_util
import pickle

class Config:
    source_dir = 'data/split/59_ae_fl_5000'
    output_dir = 'train_data/lab12'


def main():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)

    print(source_path)
    print(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_train_fn = Path().joinpath(source_path, 'all_train.npz')
    dataset_test_fn = Path().joinpath(source_path, 'all_test.npz')
    dataset_save_fn = Path().joinpath(output_path, 'model.pkl')

    print("//---- load data ----//")
    x_train, y_train_onehot, l = io_util.load_train_data_npz(dataset_train_fn, 42)
    x_test, y_test_onehot, l_test = io_util.load_train_data_npz(dataset_test_fn, 42)

    print("//----load model ----//")
    #rf = rf_util.generate_rf(x_train, y_train_onehot, x_test, y_test_onehot, 1000)

    # save
    with open(dataset_save_fn,'rb') as f:
        rf = pickle.load(f)

    print("//---- cm ----//")
    y_pred = rf_util.get_pred_y(rf, x_test, y_test_onehot, 'test')
    #CM, TP,TN
    cm = rf_util.saveCM(y_test_onehot, y_pred, l_test, output_path)
    rf_util.get_TP_TN(cm)
    y_t, y_p = rf_util.get_yPred_and_yTrue_without_one_hot(y_test_onehot, y_pred)

    print("//---- F1/Precision/Recall ----//")
    rf_util.get_F1_score(y_t, y_p)

#由於千庭的實驗是平衡資料集 

#因為最終全部類別加總的FP,FN會是相同的值
#所以多類別單算淨加總的 precision 和 recall
#值也會是相同的
    rf_util.get_precision_score(y_t, y_p)
    rf_util.get_recall_score(y_t, y_p)

    
    

if __name__ == '__main__':
    main()