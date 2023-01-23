import socket
from typing import Optional


class ChatScriptClient:
    def __init__(self, server: str, port: int):
        self._server = server
        self._port = port

    def send_and_receive_message(self, message_to_send: bytes, timeout: int = 10) -> Optional[str]:
        try:
            # Connect, send, receive and close socket. Connections are not persistent
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)  # timeout in secs
            s.connect((self._server, self._port))
            s.sendall(message_to_send)
            response = ''
            while True:
                chunk = s.recv(1024)
                if chunk == b'':
                    break
                response = response + chunk.decode("utf-8")
            s.close()
            return response
        except:
            return None
