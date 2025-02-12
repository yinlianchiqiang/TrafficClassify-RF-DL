from pathlib import Path
from utils.graphlet_utils import read_pcap, write_row
from scapy.all import IPv6


def graphletFunc(result_file, path):

    write_row(result_file, ['srcIP', 'proto', 'srcPort', 'dstPort', 'dstIP', 'len', 'time'], 'w')
    pcap_data = read_pcap(path)

    for packet in pcap_data:

        # 只看 tcp, udp
        is_tcp = packet.haslayer('TCP')
        if not is_tcp and not packet.haslayer('UDP'):
            continue

        is_ip = packet.haslayer('IP')
        if not is_ip and not packet.haslayer(IPv6):
            print(packet.summary())
            continue

        ip_info = packet['IP'] if is_ip else packet[IPv6]
        info = packet['TCP'] if is_tcp else packet['UDP']

        srcIP = ip_info.src
        proto = 'tcp' if is_tcp else 'udp'
        srcPort = info.sport
        dstPort = info.dport
        dstIP = ip_info.dst

        write_row(result_file, [srcIP, proto, srcPort, dstPort, dstIP, len(packet), packet.time])


def main():
    output_dir = 'data/preprocessed/graphlet_step_1'
    source_dir = 'data/source'

    source_path = Path(source_dir)

    print("//----start----//")
    for pcap_path in sorted(source_path.iterdir()):
        pcap_name = pcap_path.name.split('.')[0].lower()
        print('//----[' + pcap_name + "] start----//")

        if pcap_path.suffix not in ['.pcap', '.pcapng']:
            print("非 pcap, pcapng pass")
            continue

        result_file = Path().joinpath(output_dir, '{}.csv'.format(pcap_name))

        if Path.is_file(result_file):
            print("已執行 pass")
            continue

        graphletFunc(result_file, pcap_path)
        print('//----[' + pcap_name + "] end----//")
    print("//----end----//")


if __name__ == "__main__":
    main()
