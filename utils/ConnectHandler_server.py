import time
from loguru import logger
import pickle
import socket
import selectors
from tqdm import tqdm


class SocketPool:
    connections = {}
    sel = selectors.DefaultSelector()
    HOST = None
    POST = None

    @staticmethod
    def setIPAddress(HOST, POST):
        SocketPool.HOST = HOST
        SocketPool.POST = POST

    @staticmethod
    def send(conn, data, client_idx):
        binary_data = pickle.dumps(data)
        len_data = len(binary_data).to_bytes(8, byteorder="big")
        length = len(binary_data)

        binary_data = len_data + binary_data

        logger.info("sending data ({} bytes) to client#{}...", length, client_idx)
        try:
            conn.sendall(binary_data)
        except OSError as e:
            logger.error("sending failed! connection between client#{} has been closed!", client_idx)
            return False
        logger.info("sending data ({} bytes) to client#{} completely", length, client_idx)
        return True

    @staticmethod
    def receive(conn, client_idx):
        try:
            bin_len = conn.recv(8)
        except ConnectionResetError as e:
            return None

        total_length = int.from_bytes(bin_len, byteorder="big")
        if total_length == 0:
            return None

        logger.info("{}bytes data to be received from client#{}", total_length, client_idx)
        cur_length = 0
        total_data = bytes()
        pbar = tqdm(total=total_length, unit='iteration')
        while cur_length < total_length:
            data = conn.recv(min(1024000, total_length - cur_length))
            cur_length += len(data)
            total_data += data
            pbar.update(len(data))
        logger.info("receive completed")
        total_data = pickle.loads(total_data)
        return total_data

    @staticmethod
    def sendData(sc_idx, data):
        return SocketPool.send(SocketPool.connections[sc_idx][0], data, sc_idx)
        pass

    @staticmethod
    def receiveData():
        while True:
            try:
                events = SocketPool.sel.select()
            except OSError as e:
                logger.critical("All clients has disconnected!!!")
                raise ConnectionError("All clients has disconnected!!!")

            for key, mask in events:
                client_idx = key.data
                received_data = SocketPool.receive(key.fileobj, client_idx)
                if received_data is None:
                    logger.warning("client#{} disconnected", client_idx)
                    key.fileobj.close()
                    SocketPool.sel.unregister(key.fileobj)
                else:
                    return received_data, client_idx

    @staticmethod
    def register(num):
        HOST = SocketPool.HOST  #
        PORT = SocketPool.POST
        sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sc.bind((HOST, PORT))
        sc.listen(1000)
        logger.debug("waiting clients to connect...")
        count = 0
        while count < num:
            conn, addr = sc.accept()
            socketConnection = (conn, addr)
            logger.debug("client addr:{} connected", addr)
            data = SocketPool.receive(conn, addr)
            ID = data["ID"]
            logger.info("receive msg from client#{}".format(ID))
            SocketPool.connections[ID] = socketConnection
            SocketPool.sel.register(conn, selectors.EVENT_READ, ID)
            logger.debug("client#{} registered", ID)
            count += 1

        sc.setblocking(False)
        logger.debug("all clients are ready")
        return 0


class ConnectHandler(object):
    def __init__(self, num_client, HOST, POST):
        logger.debug("server boot...")
        SocketPool.setIPAddress(HOST, POST)
        self.local_data_sizes = SocketPool.register(num_client)
        self.start_time = time.time()

    def sendData(self, client_idx, data=None):
        return SocketPool.sendData(client_idx, data)

    def receiveData(self):
        return SocketPool.receiveData()
