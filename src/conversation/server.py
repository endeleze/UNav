import socket
import struct
import threading
import logging
import torch
from os.path import join, exists
from os import makedirs, mkdir
import os, json
from track import Hloc
from navigation import Trajectory,actions,command_alert,command_normal, command_debug, command_count
import numpy as np
import cv2
import jpysocket
from time import time
from datetime import datetime

class Connected_Client(threading.Thread):
    def __init__(self, parent,socket=None, address='128.122.136.173', hloc=None,trajectory=None, connections=None, destinations=None, map_scale=1,log_dir=None, logger=None,initial=False):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = len(connections)
        self.connections = connections
        self.signal = True
        self.total_connections = 0
        self.hloc = hloc
        self.trajectory=trajectory
        self.destination_coords = destinations['destination_coords']
        self.destinations = destinations["destinations"]
        self.destinations_dicts = {}
        for k0, v0 in self.destinations.items():
            building_dicts = {}
            for k1, v1 in v0.items():
                floor_dicts = {}
                for k2, v2 in v1.items():
                    list0 = []
                    for v3 in v2:
                        list0.append(list(v3.keys())[0])
                    floor_dicts.update({k2: list0})
                building_dicts.update({k1: floor_dicts})
            self.destinations_dicts.update({k0: building_dicts})
        self.log_dir = log_dir
        self.logger = logger
        self.map_scale=map_scale
        self.parent=parent

    def __str__(self):
        return str(self.id) + " " + str(self.address)

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def date(self, s):
        return [s.year, s.month, s.day, s.hour, s.minute, s.second]

    def record_user_navigation_action_paths(self, x, y, ang, path_list, action_list, destination_id, navigation_message, action_message):
        try:
            json_data = json.dumps({
                'timestamp': str(datetime.now()),
                "ux": x,
                "uy": y,
                "ang": ang,
                "path_list": path_list,
                "action_list": action_list,
                "dx": self.destination_coords[destination_id]["x"],
                "dy": self.destination_coords[destination_id]["y"],
                "navigation_msg": navigation_message,
                "action_msg": action_message,
                "client_socket_id": self.__hash__()
            })
            #with open('/home/nattachart.tak/updates-unav/mahidol-branch/navigationUpdates.txt', 'a') as file:
            with open(join(self.log_dir, 'user_navigation_action_paths.txt'), 'a') as file:
                file.write(json_data + '\n')
        except Exception as e:
            print(e)
    
    def run(self):
        while self.signal:
            # try:
            self.logger.debug("Waiting to receiver number")
            number = self.recvall(self.socket, 4)
            if not number:
                continue
            command = int.from_bytes(number, 'big')
            if command == 1:
                self.logger.info('===========Loading image===========')
                length = self.recvall(self.socket, 4)
                data = self.recvall(self.socket, int.from_bytes(length, 'big'))
                if not data:
                    continue
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                

                Destination = self.socket.recv(4096)
                Destination = jpysocket.jpydecode(Destination)
                Place, Building, Floor, Destination_ = Destination.split(',')
                dicts = self.destinations[Place][Building][Floor]
                print(dicts[0])
                for i in dicts:
                    for k, v in i.items():
                        if k == Destination_:
                            destination_id = v
                self.logger.info('=========Received one image=========')
                pose = self.hloc.get_location(img)
                image_destination = join(
                    self.log_dir, destination_id, 'images')
                if not exists(image_destination):
                    makedirs(image_destination)
                message_destination = join(
                    self.log_dir, destination_id, 'logs')
                if not exists(message_destination):
                    mkdir(message_destination)
                current_time = time()
                readable_date = datetime.fromtimestamp(current_time)
                formatted_date = readable_date.strftime('%Y-%m-%d_%H-%M-%S')

                image_num=len(self.hloc.list_2d)
                cv2.imwrite(
                    join(image_destination, formatted_date+'.png'), img)
                action_message = ''
                action_list = None
                path_list = None
                message = ''
                user_pose_x, user_pose_y, user_pose_angle = None, None, None
                if pose:
                    fail_count = 0
                    self.logger.info(f"===============================================\n Estimated location: x: %d, y: %d, ang: %d destination_id: {destination_id}\nUsed {image_num} images for localization\n===============================================" % (
                        pose[0], pose[1], pose[2]))
                    user_pose_x, user_pose_y, user_pose_angle = pose[0], pose[1], pose[2]
                    path_list=self.trajectory.calculate_path(pose[:2], destination_id)
                    if len(path_list) > 0:
                        action_list=actions(pose,path_list,self.map_scale)
                        length = action_list[0][1]
                        print(f"Total number of remaining steps: {length}")
                        for act in action_list[:5]:
                            print(act)
                        if len(action_list) != self.parent.actionlines:
                            self.parent.actionlines = len(action_list)
                            self.parent.halfway = False
                            self.parent.eighty_way = False
                            message=command_normal(action_list)
                            self.parent.base_len=length
                            action_message = "Moving"
                            print("Moving")
                            print(message)
                        elif length < 8:
                            message=command_alert(action_list)
                            action_message = "Nearby"
                            print("Nearby")
                            print(message)
                        else:
                            message=command_count(self.parent,action_list,length)
                            action_message = "Navigation steps unchanged"
                            print("Navigation steps unchanged")
                            print(message)
                            # message=command_normal(action_list)
                        message+='\n'
                    else:
                        message = 'There is no path to the destination. \n'
                else:
                    try:
                        fail_count
                    except NameError:
                        fail_count = 1
                    else:
                        fail_count += 1
                    print(f"failed {fail_count} times in a row")
                    if fail_count == 3:
                        message = "Look another direction to capturn new image to estimate new pose angle and floorplan coordinate (x,y). {failed 3 times to find the shortest path to the destination}\n"
                        fail_count = 0
                    else:
                        message = "\n"

                self.record_user_navigation_action_paths(user_pose_x, user_pose_y, user_pose_angle, path_list, action_list, destination_id, message, action_message)
                self.logger.info(f"===============================================\n {message}\n ===============================================")
                self.socket.sendall(bytes(message, 'UTF-8'))

                with open(join(message_destination, formatted_date+'.txt'), "w") as file:
                    if pose:
                        file.write(str(pose[0])+', '+str(pose[1])+'\n')
                    file.write(message)

            elif command == 0:
                self.logger.info('=====Send destination to Client=====')
                destination_dicts = str(self.destinations_dicts) + '\n'
                self.send_msg(destination_dicts)
                #self.socket.sendall(bytes(destination_dicts, 'UTF-8'))
                #print(destination_dicts)
            
            elif command == 2:
                self.logger.debug("Number 2 sent to server")
                return
            """
            except:
                self.logger.warning("Client " + str(self.address) + " has disconnected")
                self.signal = False
                self.connections.remove(self)
                break
            """
    def send_msg(self, msg):
        msg = bytes(msg, 'UTF-8')
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        self.socket.sendall(msg)

class Server():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self, root, map_data, hloc_config, server_config, host_config):
        location_config = server_config['location']
        self.log_path = join(server_config['IO_root'], 'logs', location_config['place'],
                             location_config['building'], str(location_config['floor']))
        self.scale=server_config['location']['scale']
        self.hloc = Hloc(root, map_data, hloc_config)
        self.trajectory=Trajectory(map_data)

        server_configs = server_config['server']
        host = host_config['host']
        port = server_configs['port']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(5)
        self.connections = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        self.logger.info(f'Setup server at address:{host} port:{port}')
        self.halfway = False
        self.eighty_way = False
        self.actionlines = -100

    def set_new_connections(self, map_data):
        return threading.Thread(target=self.run, args=(map_data,))


    def run(self, map_data):
        while True:
            for index in range(len(self.connections)):
                if not self.connections[index].is_alive():
                    self.logger.debug(f"Dead thread index: {index}")
            sock, address = self.sock.accept()
            self.connections.append(
                Connected_Client(parent=self,socket=sock, address=address, hloc=self.hloc,trajectory=self.trajectory, connections=self.connections,
                                 destinations=map_data, map_scale=self.scale, log_dir=self.log_path, logger=self.logger))
            self.connections[len(self.connections) - 1].start()
            self.logger.info("New connection at ID " +
                             str(self.connections[len(self.connections) - 1]))
