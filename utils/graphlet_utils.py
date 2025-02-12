from pathlib import Path
from scapy.all import PcapReader
import csv
import pickle as pk


def load(filename):
    """讀檔"""
    with open(filename, 'rb') as f:
        data = pk.load(f)
    return data

def read_pcap(path: Path):
    """讀取流量檔案"""
    packets = PcapReader(str(path))
    return packets

def write_row(fn, row_data, mode='a'):
    with open(fn, mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def add_cnt_log(log, key, cnt = 1):
    """計數器 (1層)"""
    if (key in log):
        log[key] += cnt
    else:
        log[key] = cnt

def add_cnt_log_2(log, key, val, cnt = 1):
    """計數器 (2層)"""
    if (key not in log):
        log[key] = {}
    add_cnt_log(log[key], val, cnt)


def write_log(file, log, title = '', field_name = '', split = '==='):
    """log 寫檔 (1層)"""
    fn = str(field_name)
    field_name = fn if len(fn) > 0 else title

    write_row(file, [split + title + split])
    write_row(file, [field_name, 'cnt'])
    for i in log.items():
        write_row(file, i)

def write_log_2(file, log, title = '', field_name = ''):
    """log 寫檔 (2層)"""
    write_row(file, ['===' + title +'==='])
    for key, info in log.items():
        fn = str(field_name)
        field_name = fn if len(fn) > 0 else title
        write_log(file, info, key, field_name, '---')
