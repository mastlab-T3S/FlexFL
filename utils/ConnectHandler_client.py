import time
from loguru import logger
import pickle
import socket
import selectors
from tqdm import tqdm

class ConnectHandler(object):
    def __init__(self, HOST, POST, ID):
        self.socket = None
        self.addr = (HOST, POST)
        self.ID = ID
        self.register()

    def register(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info('connected to the server...')
        self.socket.connect(self.addr)
        logger.info("connected to the server successfully")
        logger.info("sending ID to server...")
        data = {"ID": self.ID}
        self.uploadToServer(data)
        logger.info("send completed")
        logger.debug("register completed")

    def uploadToServer(self, data):
        binary_data = pickle.dumps(data)
        len_data = len(binary_data).to_bytes(8, byteorder="big")
        length = len(binary_data)

        binary_data = len_data + binary_data

        logger.info("sending data ({} bytes) to client...", length)
        self.socket.sendall(binary_data)
        logger.info("sending data ({} bytes) to client completely", length)

    def receiveFromServer(self):
        total_length = int.from_bytes(self.socket.recv(8), byteorder="big")
        if total_length == 0:
            logger.critical("connection is closed by server!!! Server may crash!!!")
        logger.info("{} bytes data to be received".format(total_length))
        cur_length = 0
        total_data = bytes()
        pbar = tqdm(total=total_length, unit='iteration')
        # self.socket.settimeout(60)
        logger.info("start recving, timeout limit: 60")
        while cur_length < total_length:
            data = self.socket.recv(min(total_length-cur_length, 1024000))
            cur_length += len(data)
            total_data += data
            pbar.update(len(data))
        logger.info("receive completed")
        total_data = pickle.loads(total_data)
        # self.socket.settimeout(None)
        logger.info("end recving, timeout limit close")
        return total_data