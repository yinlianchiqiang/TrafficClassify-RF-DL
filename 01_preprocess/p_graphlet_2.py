import argparse

from pathlib import Path
import math
import csv
from utils.constants import PREFIX_TO_TRAFFIC_ID
from utils.graphlet_networkx_utils import graphletpartite_layout
from utils.graphlet_utils import add_cnt_log, add_cnt_log_2, write_log_2, write_log, write_row
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('agg')


class Config:
    source_dir = 'data/preprocessed/graphlet_step_1'
    output_dir = 'data/preprocessed/graphlet_step_2'
    num = 20
    #isSaveImage = False
    isSaveImage = False


def getFolderFilePath(folder: Path, file_name):
    return Path().joinpath(folder, file_name)


def getFolderPath(root_path, folder):
    path = Path().joinpath(root_path, folder)
    path.mkdir(parents=True, exist_ok=True)
    return path


def graphletGraph(title, g_list, png_path, is_save=True):
    
    """ graphlet 轉為圖 (並存 png 檔) """
    DG = nx.DiGraph()

    for i in range(len(g_list)):
        for j in range(5-1):
            DG.add_node(g_list[i][j], layer=j)
            DG.add_node(g_list[i][j+1], layer=j+1)
            DG.add_edge(g_list[i][j], g_list[i][j+1], weight=1)

    # 大圖
    # subset_color = ["red", "red", "red", "red", "red", "red"]
    # n_size = 10

    # 小圖
    subset_color = ["gray", "gray", "gray", "gray", "gray", "gray"]
    n_size = 2

    color = [subset_color[data["layer"]] for v, data in DG.nodes(data=True)]
    pos = graphletpartite_layout(DG, subset_key="layer")

    # 大圖
    # plt.figure(figsize=(10, 7.5))
    # ax = plt.gca()
    # ax.set_title(title)

    # 小圖
    plt.figure(figsize=(0.5, 0.5)) #0.5,0.5 inch 0.28

    nx.draw(DG, pos, arrows=False, node_color=color, with_labels=False, node_size=n_size)

    # 取得圖片特徵
    canvas = plt.gca().figure.canvas
    canvas.draw()
    img = np.array(canvas.buffer_rgba())
    img = np.rint(img[..., :3] @ [0.2126, 0.7152, 0.0722]).astype(np.uint8)

    plt.figure()
    plt.imshow(img)
    plt.gray()
    if is_save:
        plt.imsave(png_path, img)
        # plt.savefig(png_path)

    plt.close('all')
    del canvas
    return DG, list(img.flatten() / 255)


def graphletFeature(config: Config, name, datas, folder, idx_info_str):
    len_log = {}    # 紀錄 長度 (取 log 2)
    ip_log = {}     # 紀錄 出現過的 IP (判定source ip 用)
    graphlet_sort_log = {}  # 紀錄 已排序的 graphlet (用 ";" 分割各欄位)
    graphlet_field_log = {}  # 紀錄 各欄位出現過 value (有加前贅詞)
    graphlet_field_ori_log = {}  # 紀錄 各欄位出現過 value (輸出 csv 用)
    graphlet_list = []      # 以陣列計 graphlet，一列為一筆 graphlet [srcIP, proto, srcPort, dstPort, dstIP]
    feature_log = {}    # 紀錄 feature
    feature_list = []   # 所有 feature 轉為陣列 (回傳)
    graph_feature = []

    valid_file_path = getFolderFilePath(folder, f'{idx_info_str}_valid.csv')
    png_file_path = getFolderFilePath(folder, f'{idx_info_str}.png')

    # 統計: 封包長度, IP
    for row in datas:
        add_cnt_log(ip_log, row[0])
        add_cnt_log(ip_log, row[4])
        add_cnt_log_2(len_log, row[1], getLenLog(row[5]))

    # write_log_2(valid_file_path, len_log, 'len')
    # write_log(valid_file_path, ip_log, 'ip')
    # graphlet 確認 src dst 方向
    for row in datas:
        src_ip = row[0]     # default
        src_port = row[2]   # default
        dst_port = row[3]   # default
        dst_ip = row[4]     # default
        if ip_log[src_ip] < ip_log[dst_ip] or (ip_log[src_ip] == ip_log[dst_ip] and src_ip > dst_ip):
            src_ip, dst_ip = dst_ip, src_ip
            src_port, dst_port = dst_port, src_port
        add_cnt_log(graphlet_sort_log, ';'.join([src_ip, row[1], src_port, dst_port, dst_ip]))

    # graphlet + 記錄各欄有甚麼值
    # write_row(valid_file_path, ['=== graphlet ==='])
    # write_row(valid_file_path, ['srcIP', 'proto', 'srcPort', 'dstPort', 'dstIP'])
    for log in graphlet_sort_log:
        ori_data = log.split(';')
        # write_row(valid_file_path, ori_data)

        add_cnt_log_2(graphlet_field_ori_log, 'srcIP', ori_data[0])
        add_cnt_log_2(graphlet_field_ori_log, 'proto', ori_data[1])
        add_cnt_log_2(graphlet_field_ori_log, 'srcPort', ori_data[2])
        add_cnt_log_2(graphlet_field_ori_log, 'dstPort', ori_data[3])
        add_cnt_log_2(graphlet_field_ori_log, 'dstIP', ori_data[4])

        ori_data[0] = "si_" + ori_data[0]
        ori_data[2] = "sp_" + ori_data[2]
        ori_data[3] = "dp_" + ori_data[3]
        ori_data[4] = "di_" + ori_data[4]

        add_cnt_log_2(graphlet_field_log, 'srcIP', ori_data[0])
        add_cnt_log_2(graphlet_field_log, 'proto', ori_data[1])
        add_cnt_log_2(graphlet_field_log, 'srcPort', ori_data[2])
        add_cnt_log_2(graphlet_field_log, 'dstPort', ori_data[3])
        add_cnt_log_2(graphlet_field_log, 'dstIP', ori_data[4])

        graphlet_list.append(ori_data)

    # write_log_2(valid_file_path, graphlet_field_ori_log, 'graphlet field', 'field')
    DG, graph_feature = graphletGraph(name, graphlet_list, png_file_path, config.isSaveImage)

    # feature
    col_idx = 0
    col_log = {}
    col_back_log = {}
    for col_key, col_node in graphlet_field_log.items():
        col_idx += 1

        str_out = '{}:{}'.format(col_idx, (col_idx+1))
        str_in = '{}:{}'.format(col_idx, (col_idx-1))

        add_cnt_log_2(feature_log, 'n', col_idx, len(col_node))

        for node_key, node_val in col_node.items():
            out_deg = 0
            in_deg = 0

            if col_idx != 1:
                in_deg = DG.in_degree(node_key)
                add_cnt_log_2(col_log, str_in, node_key, in_deg)
            if col_idx != 5:
                out_deg = DG.out_degree(node_key)
                add_cnt_log_2(col_log, str_out, node_key, out_deg)
            if col_idx != 1 and col_idx != 5:
                add_cnt_log_2(col_back_log, str_in, node_key, out_deg)
                add_cnt_log_2(col_back_log, str_out, node_key, in_deg)

    for d_key, d_data in col_log.items():
        tmp_a = max(d_data.items(), key=lambda x: x[1])
        tmp_o = [d for i, d in d_data.items() if d == 1]
        tmp_s = sum(d_data.values())

        add_cnt_log_2(feature_log, 'a', d_key, tmp_a[1])
        add_cnt_log_2(feature_log, 'o', d_key, len(tmp_o))
        add_cnt_log_2(feature_log, 'u', d_key, tmp_s/len(d_data))

        if (d_key in col_back_log):
            tmp_b = col_back_log[d_key][tmp_a[0]]
            add_cnt_log_2(feature_log, 'b', d_key, tmp_b)

    for key, data in feature_log.items():
        feature_list = feature_list + [d for i, d in data.items()]

    
    tcp_len = 0
    udp_len = 0
    for i in range(12):
        _tmp_l = len_log['tcp'][i] if 'tcp' in len_log and i in len_log['tcp'] else 0
        tcp_len = tcp_len + _tmp_l
    for i in range(12):
        _tmp_l = len_log['udp'][i] if 'udp' in len_log and i in len_log['udp'] else 0
        udp_len = udp_len + _tmp_l

    for i in range(12):
        if (tcp_len > 0):
            tmp_l = len_log['tcp'][i] if 'tcp' in len_log and i in len_log['tcp'] else 0
            add_cnt_log_2(feature_log, 'l_tcp', i, tmp_l)
            tmp_l = float(tmp_l)/tcp_len
            feature_list.append(tmp_l)
        else: 
            add_cnt_log_2(feature_log, 'l_tcp', i, 0)
            feature_list.append(0)

    for i in range(12):
        if (udp_len > 0):
            tmp_l = len_log['udp'][i] if 'udp' in len_log and i in len_log['udp'] else 0
            add_cnt_log_2(feature_log, 'l_udp', i, tmp_l)
            tmp_l = float(tmp_l)/udp_len
            feature_list.append(tmp_l)
        else: 
            add_cnt_log_2(feature_log, 'l_udp', i, 0)
            feature_list.append(0)

    # write_log_2(valid_file_path, feature_log, 'feature', 'field')

    return feature_list, graph_feature


def getLenLog(str):
    result = 0
    len = int(str)
    if len - 1024 >= 0:
        result = 11
    else:
        result = int(math.ceil(math.log(len, 2)))
    return result


def saveNpz(datas, prefix, label, idx, folder):
    npz_dir = getFolderPath(Config.output_dir, f'npz/{folder}')
    cond_name = getFolderFilePath(npz_dir, f'{prefix}_part_{idx}.npz')
    # cond_name = os.path.join(npz_dir,  '{}_part_{}.npz'.format(prefix, idx))
    y = [label] * len(datas)
    np.savez_compressed(cond_name, data=np.array(datas), y=y)
    del y
    print(f'{folder} - Done')


def preprocess(config: Config, pcap_path: Path, output_path: Path):
    pcap_name = pcap_path.name.split('.')[0].lower()
    traffic_id = PREFIX_TO_TRAFFIC_ID.get(pcap_name)
    data_folder = getFolderPath(output_path, pcap_name)
    check_folder = getFolderPath(Config.output_dir, f'npz/2559')
    feature_file = getFolderFilePath(output_path, f'{pcap_name}.csv')

    # 確認沒有 label
    if traffic_id is None:
        print('not in class id')
        return

    # # 確認是否重複做
    # if Path.is_file(feature_file):
    #     print("duplicate")
    #     return

    features_ori = []
    features_graph = []
    feature_index = 0
    datas = []  # 一個 graphlet 的 datas
    batch_index = 0
    batch_size = 10000
    with open(pcap_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        isBatchCheck = False
        isBatchContinue = False
        check_fn = f'{pcap_name}_part_{0}.npz'
        check_fpath = getFolderFilePath(check_folder, check_fn)
        
        for row in itertools.islice(reader, 1, None):
            
            if (len(datas) >= config.num):
                datas = datas[1:config.num]
            datas.append(row)

            if len(datas) < config.num:
                continue

            if isBatchCheck == False:
                isBatchCheck = True
                isBatchContinue = Path.is_file(check_fpath)
                if isBatchContinue:
                    print(f'[{pcap_name} - Round {batch_index}] has been completed')
                else:
                    print(f'[{pcap_name} - Round {batch_index}] - Start')

            if isBatchContinue == False:
                f, g = graphletFeature(config, pcap_name, datas, data_folder, feature_index)
                tmp_g = list(f)
                # write_row(feature_file, [traffic_id, tmp_g])
                tmp_g.extend(g)
                features_ori.append(f)
                features_graph.append(tmp_g)
            feature_index += 1

            if feature_index > 0 and feature_index % batch_size == 0:
                if isBatchContinue == False:
                    print(f'len = {feature_index}, batch len = {len(features_ori)}')
                    if (len(features_ori) != len(features_graph)):
                        print("ori len != graph len")

                    saveNpz(features_ori, pcap_name, traffic_id, batch_index, '59')
                    saveNpz(features_graph, pcap_name, traffic_id, batch_index, '2559')
                    print(f'[{pcap_name} - Round {batch_index}] - End')


                batch_index += 1
                isBatchCheck = False
                isBatchContinue = False
                check_fn = f'{pcap_name}_part_{batch_index}.npz'
                check_fpath = getFolderFilePath(check_folder, check_fn)
                features_ori.clear()
                features_graph.clear()

        if features_ori:
            if Path.is_file(check_fpath):
                print(f'[{pcap_name} - Round {batch_index}] has been completed')
            else:
                print(f'len = {feature_index}, batch len = {len(features_ori)}')
                if (len(features_ori) != len(features_graph)):
                    print("ori len != graph len")

                saveNpz(features_ori, pcap_name, traffic_id, batch_index, '59')
                saveNpz(features_graph, pcap_name, traffic_id, batch_index, '2559')
                print(f'[{pcap_name} - Round {batch_index}] - End')

                batch_index += 1
                features_ori.clear()
                features_graph.clear()


def main():
    config = Config()

    source_path = Path(config.source_dir)
    output_path = Path(config.output_dir)

    print("//----start----//")

    for pcap_path in sorted(source_path.iterdir()):
        print('//----[' + pcap_path.name + "] start----//")
        preprocess(config, pcap_path, output_path)
        print('//----[' + pcap_path.name + "] end----//")

    print("//----end----//")


if __name__ == "__main__":
    main()
