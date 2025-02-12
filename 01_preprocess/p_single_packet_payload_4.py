import os
from pathlib import Path

import numpy as np
from scapy.packet import Raw
# https://github.com/phaethon/kamene/issues/226
# from scapy.utils import PcapReader
from scapy.all import PcapReader

from utils.constants import PREFIX_TO_TRAFFIC_ID

output_dir = '00_data/preprocessed/[Name Yourself]'
source_dir = '00_data/source'


def read_pcap(path: Path):
    """讀取流量檔案"""
    packets = PcapReader(str(path))
    return packets


def payload_to_npy(payload, max_len=1024):
    """將 payload 轉成 npy 陣列"""
    img_width = int(max_len ** 0.5)
    hexst = payload.hex()   # 轉為 16 進位 (4bit)
    arr = np.array([int(hexst[i], 16) for i in range(0, len(hexst))])
    arr = arr[0: max_len]   # 取前 max_len 個截斷
    arr = np.pad(arr, (0, max_len-len(arr)), 'constant', constant_values=0)  # 補 0
    result = np.reshape(arr, (-1, img_width))   # 轉為 2 維 (類似圖片)
    result = np.uint8(result) / 15  # 標準化成 [0..1]
    return result


def get_data_by_file(path):
    """取得檔案中各封包 npy 陣列"""
    result = []
    for i, packet in enumerate(read_pcap(path)):
        if not packet.haslayer(Raw):
            continue
        payload = packet[Raw].load
        arr = payload_to_npy(payload)
        result.append(arr)
    result = np.array(result)
    print(result.shape)
    return result


def preprocess(path: Path, cnt):
    tail = path.name
    prefix = path.name.split('.')[0].lower()
    class_id = PREFIX_TO_TRAFFIC_ID.get(prefix)

    print('--- [' + tail + ' Start ]---')

    # 確認沒有 label
    if class_id is None:
        print('not in class id')
        return

    # 確認是否重複做
    check_name = os.path.join(output_dir,  '{}_part_{}.npz'.format(prefix, 0))
    if os.path.isfile(check_name):
        print('--- pass ---')
        return

    # 取得檔案中各封包 npy 陣列
    datas = []

    data_index = 0
    batch_index = 0
    batch_size = 10000
    for packet in read_pcap(path):
        if not packet.haslayer(Raw):
            continue
        payload = packet[Raw].load
        datas.append(payload_to_npy(payload))
        data_index += 1
        if datas and data_index > 0 and data_index % batch_size == 0:
            saveNpz(datas, prefix, class_id, batch_index)
            batch_index += 1
            datas.clear()
    if datas:
        saveNpz(datas, prefix, class_id, batch_index)
        datas.clear()

    cnt[prefix] = data_index
    print('len = ' + str(data_index))

    print('--- [' + tail + ' End ]---')
    return 'Done'


def saveNpz(datas, prefix, label, idx):
    cond_name = os.path.join(output_dir,  '{}_part_{}.npz'.format(prefix, idx))
    y = [label] * len(datas)
    np.savez_compressed(cond_name, data=np.array(datas), y=y)
    del y
    print('Round ' + str(idx) + ' - Done')


def main():
    source_path = Path(source_dir)

    cnt = dict()
    for pcap_path in sorted(source_path.iterdir()):
        preprocess(pcap_path, cnt)

    print(cnt)
    print('OVER')


if __name__ == '__main__':
    main()
