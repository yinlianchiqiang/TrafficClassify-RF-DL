from pathlib import Path

from copy import deepcopy

import utils.io_util as io_util
import utils.rf_util as rf_util

class Config:
    source_dir = 'data/split/59/1d'
    output_dir = 'train_data/lab11'
    client_num = 2
    server_tree_num = 1000
    client_update_tree_num = 500
    
def getClientData(client_num, source_path):
  """取得各 Client 訓練資料"""
  result = []
  for idx in range(client_num):
    train_path = Path().joinpath(source_path, f'c{idx}_train.npz')
    val_path = Path().joinpath(source_path, f'c{idx}_val.npz')
    print(f"Client load data - {idx}")
    x_train, y_train, l_train = io_util.load_train_data_npz(train_path, 42)
    x_val, y_val, l_val = io_util.load_train_data_npz(val_path, 42)

    result.append({
       'x_train': x_train, 'y_train': y_train, 'l_train': l_train,
       'x_val': x_val, 'y_val': y_val, 'l_val': l_val
    })
  return result

def getClientRF(client_data):
  """取得各 Client 訓練完的隨機森林"""
  _result = []
  _tree_cnt = 1000 // len(client_data)
  print('tree count:', _tree_cnt)
  for idx, _d in enumerate(client_data):
    print(f"Client train RF - {idx}")
    _rf = rf_util.generate_rf(_d['x_train'], _d['y_train'], _d['x_val'], _d['y_val'], _tree_cnt)
    _result.append(_rf)
  return _result

def getDTAccs(_rf, rf_idx, val_X, val_y):
  """取得隨機森林中決策樹的正確率及排序"""
  accs = []
  for idx, tree in enumerate(_rf.estimators_):
    tree.classes_ = _rf.classes_

    accs.append((rf_idx, idx, rf_util.get_acc(tree, val_X, val_y)))
  accs = sorted(accs, key=lambda x: x[2], reverse = True)
  return accs

def getClientRFAccs(c_rfs, data):
  """取得各 Client 的隨機森林中決策樹的正確率及排序"""
  _result = []
  for idx, c_rf in enumerate(c_rfs):
    _accs = getDTAccs(c_rf, idx, data[idx]['x_val'], data[idx]['y_val'])
    _result.append(_accs)
  return _result

def getServerRank(k, c_rf_accs):
  """Server 從各 Client 收集 k 棵樹"""
  global_accs = []
  for c_rf_acc in c_rf_accs:
    global_accs = global_accs + c_rf_acc[0:k]
  global_accs = sorted(global_accs, key=lambda x: x[2], reverse = True)
  return global_accs

def getServerRF(T, global_accs, c_rfs):
  """Server 取正確率最高的 T 棵決策樹，組成 Server 隨機森林"""
  # global_accs，為 Server 收到的樹的正確率
  # 防止取超過收到的數量
  if len(global_accs) < T:
    print(f'max len = {len(global_accs)}')
    T = len(global_accs)
    
  forest = []
  for idx in range(T):
    rank_info = global_accs[idx]  # rank_info - 0: Client idx, 1: tree 在原本模型的 idx
    forest.append(c_rfs[rank_info[0]].estimators_[rank_info[1]])
  tmp_rf = deepcopy(c_rfs[1])
  tmp_rf.estimators_ = forest
  tmp_rf.n_estimators = len(tmp_rf.estimators_)
  return tmp_rf

def main():
    source_path = Path(Config.source_dir)
    output_path = Path(Config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("//---- load data ----//")
    client_data = getClientData(Config.client_num, source_path)

    print("//---- start train ----//")
    c_rfs = getClientRF(client_data)

    print("//---- calc client RF DTs Acc ----//")
    c_rf_accs = getClientRFAccs(c_rfs, client_data)

    print("//---- Server get Tree ----//")
    global_accs = getServerRank(Config.client_update_tree_num, c_rf_accs)
    server_rf = getServerRF(Config.server_tree_num, global_accs, c_rfs)

    print("//---- Server test cm ----//")
    test_fn = Path().joinpath(source_path, 'all_test.npz')
    x_test, y_test_onehot, l_test = io_util.load_train_data_npz(test_fn, 42)
    y_pred = rf_util.get_pred_y(server_rf, x_test, y_test_onehot, 'server')
    cm = rf_util.saveCM(y_test_onehot, y_pred, l_test, output_path)


    rf_util.get_TP_TN(cm)
    y_t, y_p = rf_util.get_yPred_and_yTrue_without_one_hot(y_test_onehot, y_pred)

    print("//---- F1/Precision/Recall ----//")
    rf_util.get_F1_score(y_t, y_p)



if __name__ == '__main__':
    main()