import math
from pathlib import Path

import numpy as np

from utils.constants import ID_TO_TRAFFIC, PREFIX_TO_TRAFFIC_ID


class Config:
    source_dir = 'data/preprocessed/choose/2559'
    output_dir = 'data/split/2559/1d'
    mode = '1d'  # [1d/2d]
    max_len = 44865 #單封包 68594 多封包 44865
    client_num = 2
    data_shape = 2559


def load_data_1d(config: Config, num_setting):
    npz_max_len = 10000
    train_rate = 0.80
    val_rate = 0.10

    data_shape = config.data_shape

    x_train = np.empty((0, data_shape))
    y_train = np.empty((0))
    x_val = np.empty((0, data_shape))
    y_val = np.empty((0))
    x_test = np.empty((0, data_shape))
    y_test = np.empty((0))

    x_c_train = []
    y_c_train = []
    x_c_val = []
    y_c_val = []

    for idx in range(0, config.client_num):
        x_c_train.append(np.empty((0, data_shape)))
        y_c_train.append(np.empty((0)))
        x_c_val.append(np.empty((0, data_shape)))
        y_c_val.append(np.empty((0)))

    for label, l_info in num_setting.items():
        print("====== [" + label + " - start] ======")
        x_label = np.empty((0, data_shape))
        y_label = np.empty((0))

        # 取得各檔資料
        for prefix, p_cnt in l_info['prefixs'].items():
            # print("------ [" + prefix + " - start] ------")
            file_cnt = math.ceil(p_cnt / npz_max_len)

            x_prefix = np.empty((0, data_shape))
            y_prefix = np.empty((0))

            for f_idx in range(0, file_cnt):
                file_name = Path().joinpath(config.source_dir, f"{prefix}_part_{f_idx}.npz")

                # 確認檔案是否存在
                if not Path.is_file(file_name):
                    print('檔案不存在' + str(file_name))
                    return
                # print("--- [" + file_name.name + " - start] ---")

                tmp_data = np.load(file_name, allow_pickle=True)
                tmp_x = tmp_data['data']
                tmp_y = tmp_data['y']

                if f_idx == (file_cnt - 1):
                    x_prefix = np.concatenate((x_prefix, tmp_x[:p_cnt % npz_max_len, :]), axis=0)
                    y_prefix = np.concatenate((y_prefix, tmp_y[:p_cnt % npz_max_len]), axis=0)
                else:
                    x_prefix = np.concatenate((x_prefix, tmp_x), axis=0)
                    y_prefix = np.concatenate((y_prefix, tmp_y), axis=0)
                # print("--- [" + file_name.name + " - end] ---")

            x_label = np.concatenate((x_label, x_prefix), axis=0)
            y_label = np.concatenate((y_label, y_prefix), axis=0)
            # print("------ [" + prefix + " - end] ------")
        print('x_label - data shape', x_label.shape)
        print('y_label - y shape', y_label.shape)

        # 集中式資料集劃分 => [ train             || val           || test ]
        # 聯合式資料集劃分 => [ train 1 | train 2 || val 1 | val 2 || test ]

        print(x_label.shape[0])
        train_num = int(x_label.shape[0] * train_rate)
        val_num = int(x_label.shape[0] * val_rate)

        if config.client_num != 0:
            train_partition = train_num // (config.client_num)
            val_partition = val_num // (config.client_num)

            for idx in range(0, config.client_num):
                train_start = idx * train_partition
                train_end = (idx + 1) * train_partition
                val_start = train_num + idx * val_partition
                val_end = train_num + (idx + 1) * val_partition

                x_c_train[idx] = np.concatenate((x_c_train[idx], x_label[train_start: train_end, :]), axis=0)
                y_c_train[idx] = np.concatenate((y_c_train[idx], y_label[train_start: train_end]), axis=0)
                x_c_val[idx] = np.concatenate((x_c_val[idx], x_label[val_start: val_end, :]), axis=0)
                y_c_val[idx] = np.concatenate((y_c_val[idx], y_label[val_start: val_end]), axis=0)

                print(f"Client {idx} - train: {train_start} ~ {train_end}")
                print(f"Client {idx} - val: {val_start} ~ {val_end}")

        val_max = train_num + val_num

        x_train = np.concatenate((x_train, x_label[:train_num, :]), axis=0)
        y_train = np.concatenate((y_train, y_label[:train_num]), axis=0)
        x_val = np.concatenate((x_val, x_label[train_num: val_max, :]), axis=0)
        y_val = np.concatenate((y_val, y_label[train_num: val_max]), axis=0)
        x_test = np.concatenate((x_test, x_label[val_max:, :]), axis=0)
        y_test = np.concatenate((y_test, y_label[val_max:]), axis=0)

        print(f"ALL - train: 0 ~ {train_num}")
        print(f"ALL - val: {train_num} ~ {val_max}")
        print(f"ALL - test: {val_max} ~ {x_label.shape[0]}")

    print("==========================================================================")

    print(f"ALL - train shape - x: {x_train.shape}")
    print(f"ALL - train shape - y: {y_train.shape}")
    print(f"ALL - val shape - x: {x_val.shape}")
    print(f"ALL - val shape - y: {y_val.shape}")
    print(f"ALL - test shape - x: {x_test.shape}")
    print(f"ALL - test shape - y: {y_test.shape}")

    saveNpz(config.output_dir, 'all_train', x_train, y_train)
    saveNpz(config.output_dir, 'all_val', x_val, y_val)
    saveNpz(config.output_dir, 'all_test', x_test, y_test)

    for idx in range(0, config.client_num):
        tmpStr = str(idx)
        saveNpz(config.output_dir, f'c{tmpStr}_train', x_c_train[idx], y_c_train[idx])
        saveNpz(config.output_dir, f'c{tmpStr}_val', x_c_val[idx], y_c_val[idx])
        print(f"Client {idx} - train shape - x: {x_c_train[idx].shape}")
        print(f"Client {idx} - train shape - y: {y_c_train[idx].shape}")
        print(f"Client {idx} - val shape - x: {x_c_val[idx].shape}")
        print(f"Client {idx} - val shape - y: {y_c_val[idx].shape}")


def load_data_2d(config: Config, num_setting):
    npz_max_len = 10000
    train_rate = 0.80
    val_rate = 0.10

    data_shape = 32

    x_train = np.empty((0, data_shape, data_shape))
    y_train = np.empty((0))
    x_val = np.empty((0, data_shape, data_shape))
    y_val = np.empty((0))
    x_test = np.empty((0, data_shape, data_shape))
    y_test = np.empty((0))

    x_c_train = []
    y_c_train = []
    x_c_val = []
    y_c_val = []

    for idx in range(0, config.client_num):
        x_c_train.append(np.empty((0, data_shape, data_shape)))
        y_c_train.append(np.empty((0)))
        x_c_val.append(np.empty((0, data_shape, data_shape)))
        y_c_val.append(np.empty((0)))

    for label, l_info in num_setting.items():
        print("====== [" + label + " - start] ======")
        x_label = np.empty((0, data_shape, data_shape))
        y_label = np.empty((0))

        # 取得各檔資料
        for prefix, p_cnt in l_info['prefixs'].items():
            # print("------ [" + prefix + " - start] ------")
            file_cnt = math.ceil(p_cnt / npz_max_len)

            x_prefix = np.empty((0, data_shape, data_shape))
            y_prefix = np.empty((0))

            for f_idx in range(0, file_cnt):
                file_name = Path().joinpath(config.source_dir, f"{prefix}_part_{f_idx}.npz")

                # 確認檔案是否存在
                if not Path.is_file(file_name):
                    print('檔案不存在' + str(file_name))
                    return
                # print("--- [" + file_name.name + " - start] ---")

                tmp_data = np.load(file_name, allow_pickle=True)
                tmp_x = tmp_data['data']
                tmp_y = tmp_data['y']
                
                print("x_prefix shape:", x_prefix.shape)
                print("tmp_x slice shape:", tmp_x[:p_cnt % npz_max_len, :].shape)

                if f_idx == (file_cnt - 1):
                    x_prefix = np.concatenate((x_prefix, tmp_x[:p_cnt % npz_max_len, :]), axis=0)
                    y_prefix = np.concatenate((y_prefix, tmp_y[:p_cnt % npz_max_len]), axis=0)
                else:
                    x_prefix = np.concatenate((x_prefix, tmp_x), axis=0)
                    y_prefix = np.concatenate((y_prefix, tmp_y), axis=0)
                # print("--- [" + file_name.name + " - end] ---")

            x_label = np.concatenate((x_label, x_prefix), axis=0)
            y_label = np.concatenate((y_label, y_prefix), axis=0)
            # print("------ [" + prefix + " - end] ------")
        print('x_label - data shape', x_label.shape)
        print('y_label - y shape', y_label.shape)

        # 集中式資料集劃分 => [ train             || val           || test ]
        # 聯合式資料集劃分 => [ train 1 | train 2 || val 1 | val 2 || test ]

        print(x_label.shape[0])
        train_num = int(x_label.shape[0] * train_rate)
        val_num = int(x_label.shape[0] * val_rate)

        if config.client_num != 0:
            train_partition = train_num // (config.client_num)
            val_partition = val_num // (config.client_num)

            for idx in range(0, config.client_num):
                train_start = idx * train_partition
                train_end = (idx + 1) * train_partition
                val_start = train_num + idx * val_partition
                val_end = train_num + (idx + 1) * val_partition

                x_c_train[idx] = np.concatenate((x_c_train[idx], x_label[train_start: train_end, :]), axis=0)
                y_c_train[idx] = np.concatenate((y_c_train[idx], y_label[train_start: train_end]), axis=0)
                x_c_val[idx] = np.concatenate((x_c_val[idx], x_label[val_start: val_end, :]), axis=0)
                y_c_val[idx] = np.concatenate((y_c_val[idx], y_label[val_start: val_end]), axis=0)

                print(f"Client {idx} - train: {train_start} ~ {train_end}")
                print(f"Client {idx} - val: {val_start} ~ {val_end}")

        val_max = train_num + val_num

        x_train = np.concatenate((x_train, x_label[:train_num, :]), axis=0)
        y_train = np.concatenate((y_train, y_label[:train_num]), axis=0)
        x_val = np.concatenate((x_val, x_label[train_num: val_max, :]), axis=0)
        y_val = np.concatenate((y_val, y_label[train_num: val_max]), axis=0)
        x_test = np.concatenate((x_test, x_label[val_max:, :]), axis=0)
        y_test = np.concatenate((y_test, y_label[val_max:]), axis=0)

        print(f"ALL - train: 0 ~ {train_num}")
        print(f"ALL - val: {train_num} ~ {val_max}")
        print(f"ALL - test: {val_max} ~ {x_label.shape[0]}")

    print("==========================================================================")

    print(f"ALL - train shape - x: {x_train.shape}")
    print(f"ALL - train shape - y: {y_train.shape}")
    print(f"ALL - val shape - x: {x_val.shape}")
    print(f"ALL - val shape - y: {y_val.shape}")
    print(f"ALL - test shape - x: {x_test.shape}")
    print(f"ALL - test shape - y: {y_test.shape}")

    saveNpz(config.output_dir, 'all_train', x_train, y_train)
    saveNpz(config.output_dir, 'all_val', x_val, y_val)
    saveNpz(config.output_dir, 'all_test', x_test, y_test)

    for idx in range(0, config.client_num):
        tmpStr = str(idx)
        saveNpz(config.output_dir, f'c{tmpStr}_train', x_c_train[idx], y_c_train[idx])
        saveNpz(config.output_dir, f'c{tmpStr}_val', x_c_val[idx], y_c_val[idx])
        print(f"Client {idx} - train shape - x: {x_c_train[idx].shape}")
        print(f"Client {idx} - train shape - y: {y_c_train[idx].shape}")
        print(f"Client {idx} - val shape - x: {x_c_val[idx].shape}")
        print(f"Client {idx} - val shape - y: {y_c_val[idx].shape}")


def saveNpz(folder, name, x, y):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    cond_name = Path().joinpath(folder_path,  f'{name}.npz')
    np.savez_compressed(cond_name, x=x, y=y)


def get_data_count(folder):
    p = Path(folder).glob('**/*.npz')
    files = [x.stem for x in p if x.is_file()]
    files_count = {}
    data_count = {}
    batch_size = 10000

    for fn in files:
        prefix = fn.split('_part_')[0]
        if (prefix not in files_count):
            files_count[prefix] = 0
        files_count[prefix] += 1
    for prefix, num in files_count.items():
        final_idx = num - 1
        fn = Path().joinpath(folder, f'{prefix}_part_{final_idx}.npz')
        data = np.load(fn, allow_pickle=True)
        y = data['y']
        data_count[prefix] = batch_size * final_idx + y.shape[0]

    print('data_count', data_count)
    return data_count


def getNumSetting(folder, max_len=100000):
    counter = {}
    num_setting = {}
    data_count = get_data_count(folder)

    # 先統計各類別
    for prefix, cnt in data_count.items():
        traffic_id = PREFIX_TO_TRAFFIC_ID.get(prefix)
        traffic_name = ID_TO_TRAFFIC[traffic_id]

        if (traffic_name not in counter):
            counter[traffic_name] = {'prefixs': {}, 'cnt': 0}
        counter[traffic_name]['prefixs'][prefix] = cnt
        counter[traffic_name]['cnt'] += cnt
    # print('counter', counter)

    # 確定各類別要取多少資料
    # min_data = min(counter.items(), key=lambda x: x[1]['cnt'])
    # min_cnt = min_data[1]['cnt']
    # max_len = min_cnt if min_cnt <= max_len else max_len
    # print('label_max_len', max_len)

    # 確定各類別的各檔要取多少資料
    for label, l_info in counter.items():
        l_max_len = l_info['cnt'] if l_info['cnt'] <= max_len else max_len
        num_setting[label] = {'prefixs': {}, 'cnt': 0}
        prefix_cnt = len(l_info['prefixs'])
        prefix_num = l_max_len // prefix_cnt  # default，理想情況下所有檔案要提供的量，餘數給資料量最多的檔案負擔

        print('['+label+']max_len', l_max_len)

        p_idx = 1
        left_len = l_max_len
        for prefix, p_cnt in sorted(l_info['prefixs'].items(), key=lambda x: x[1]):
            tmp_prefixs = num_setting[label]['prefixs']
            if prefix_num > p_cnt:  # 該檔不能負擔分攤到的量 - 取全部，並更新剩下檔案需要分攤的量
                tmp_prefixs[prefix] = p_cnt
                if p_idx < prefix_cnt:
                    prefix_num = (left_len - p_cnt) // (prefix_cnt - p_idx)
            else:                   # 該檔能負擔分攤到的量 - 資料量最多的檔案負擔餘數
                if p_idx < prefix_cnt:
                    tmp_prefixs[prefix] = prefix_num
                else:
                    tmp_prefixs[prefix] = l_max_len - num_setting[label]['cnt']
            left_len -= tmp_prefixs[prefix]
            num_setting[label]['cnt'] += tmp_prefixs[prefix]
            p_idx += 1
    print('num_setting', num_setting)
    return num_setting


def main():
    config = Config()

    print(config)

    num_setting = getNumSetting(config.source_dir, config.max_len)
    if config.mode == '1d':
        load_data_1d(config, num_setting)
    elif config.mode == '2d':
        load_data_2d(config, num_setting)

    print('OVER')


if __name__ == '__main__':
    main()
