import argparse
from os.path import dirname,join,exists,realpath
import yaml
import socket
from conversation import Server,Connected_Client,utils
import loader

def get_server(root,map_data,hloc_config,server_config, host_config):
    # IFACE = utils.get_wireless_iface()
    # return Server(server_ip_address=server_config['host'], server_port=server_config['port'], 
    #             feedback_freq=server_config['feedback_freq'], iface=IFACE, fec=server_config['FEC'], debug=server_config['DEBUG'], infer=server_config['INFERENCE'], encoding=server_config['encoding'], sink_type=server_config['sink_type'])
    return Server(root,map_data,hloc_config,server_config,host_config)

def main(root,hloc_config,server_config,host_config):
    map_data=loader.load_data(server_config)
    server = get_server(root,map_data,hloc_config,server_config,host_config)
    newConnectionsThread=server.set_new_connections(map_data)
    newConnectionsThread.start()

if __name__=='__main__':
    root = dirname(realpath(__file__)).replace('/src','')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server_config', type=str, default='configs/server/server.yaml')
    parser.add_argument('-l', '--hloc_config', type=str, default='configs/hloc.yaml')
    parser.add_argument('-c', '--host_config', type=str, default='configs/host.yaml')
    args = parser.parse_args()
    with open(args.hloc_config, 'r') as f:
        hloc_config = yaml.safe_load(f)
    with open(args.server_config, 'r') as f:
        server_config = yaml.safe_load(f)
    with open(args.host_config, 'r') as f:
        host_config = yaml.safe_load(f)
    main(root,hloc_config,host_config)
