import os

from pathlib import Path
import numpy as np
# https://github.com/phaethon/kamene/issues/226
# from scapy.utils import PcapReader
from scapy.all import PcapReader
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
#from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scapy.packet import Padding
from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Raw

from utils.constants import PREFIX_TO_TRAFFIC_ID


class Config:
    output_dir = 'data/preprocessed/single_1d_1500'
    source_dir = 'data/source'
    mode = 'all'  # all/payload
    packet_len = 1500


def read_pcap(path: Path):
    """讀取流量檔案"""
    packets = PcapReader(str(path))
    return packets


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False

# ===== all packet process start  ===== #


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"
    #elif IPv6 in packet:
    #    packet[IPv6].src = "0:0:0:0:0:0:0:0"
    #    packet[IPv6].dst = "0:0:0:0:0:0:0:0"

    return packet


def mask_port(packet):
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0

    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0
    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet
# ===== all packet process end ===== #


def packet_to_array_1d(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    # FIXME: 有空優化
    # arr = sparse.csr_matrix(arr)
    return arr


def packet_to_array_2d(packet, max_length=1024):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    img_width = int(max_length ** 0.5)

    # 補 0
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    arr = np.reshape(arr, (-1, img_width))   # 轉為 2 維 (類似圖片)

    # FIXME: 有空優化
    # arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet, config):
    if should_omit_packet(packet):
        return None

    if config.mode == 'all':
        packet = remove_ether_header(packet)
        packet = pad_udp(packet)
        packet = mask_ip(packet)
        packet = mask_port(packet)
        arr = packet_to_array_1d(packet, config.packet_len)  # 1500
        return arr
    elif config.mode == 'payload':
        if not packet.haslayer(Raw):
            return None
        payload = packet[Raw].load
        arr = packet_to_array_2d(payload, 1024)  # 1024=32*32
        return arr


def preprocess(path: Path, cnt, config: Config):
    tail = path.name
    prefix = path.name.split('.')[0].lower()
    class_id = PREFIX_TO_TRAFFIC_ID.get(prefix)

    print('--- [' + tail + ' Start ]---')

    # 確認沒有 label
    if class_id is None:
        print('not in class id')
        return

    # 確認是否重複做
    check_name = os.path.join(config.output_dir,  '{}_part_{}.npz'.format(prefix, 0))
    if os.path.isfile(check_name):
        print('duplicate')
        return

    # 取得檔案中各封包 npy 陣列
    datas = []

    data_index = 0
    batch_index = 0
    batch_size = 10000
    for packet in read_pcap(path):
        arr = transform_packet(packet, config)
        if arr is None:
            continue
        datas.append(arr)
        data_index += 1
        if datas and data_index > 0 and data_index % batch_size == 0:
            saveNpz(datas, config.output_dir, prefix, class_id, batch_index)
            batch_index += 1
            datas.clear()

    if datas:
        saveNpz(datas, config.output_dir, prefix, class_id, batch_index)
        datas.clear()

    cnt[prefix] = data_index
    print('len = ' + str(data_index))

    print('--- [' + tail + ' End ]---')
    return 'Done'


def saveNpz(datas, folder, prefix, label, idx):
    cond_name = os.path.join(folder,  '{}_part_{}.npz'.format(prefix, idx))
    y = [label] * len(datas)
    np.savez_compressed(cond_name, data=np.array(datas), y=y)
    del y
    print('Round ' + str(idx) + ' - Done')


def main():
    config = Config()

    source_path = Path(config.source_dir)
    output_path = Path(Config.output_dir).mkdir(parents=True, exist_ok=True)

    cnt = dict()
    for pcap_path in sorted(source_path.iterdir()):
        preprocess(pcap_path, cnt, config)

    print(cnt)
    print('OVER')


if __name__ == '__main__':
    main()
