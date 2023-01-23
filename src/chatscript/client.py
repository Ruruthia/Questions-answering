import socket

import click


FALLBACK_MESSAGE = "fallback"


def send_and_receive_message(message_to_send, server='127.0.0.1', port=1024, timeout=10):
    try:
        # Connect, send, receive and close socket. Connections are not persistent
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)  # timeout in secs
        s.connect((server, port))
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


@click.command()
@click.option('--name', type=click.STRING)
@click.option('--server', type=click.STRING, default='127.0.0.1')
@click.option('--port', type=click.INT, default=1024)
@click.option('--bot', type=click.STRING, default='CriStian')
def main(name, server, port, bot):
    print("Hi " + name + ", enter ':quit' to end this session")

    s = input("[" + name + "]" + ">: ").lower().strip()
    while s != ':quit':
        # Ensure empty strings are padded with at least one space before sending to the
        # server, as per the required protocol
        if s == "":
            s = " "
        # Send this to the server and print the response
        # Put in null terminations as required
        msg = u'%s\u0000%s\u0000%s\u0000' % (name, bot, s)
        msg = str.encode(msg)
        resp = send_and_receive_message(msg, server=server, port=port)

        if resp is None:
            raise RuntimeError("Error communicating with Chat Server")

        elif resp == FALLBACK_MESSAGE:
            # TODO
            pass

        else:
            print("[Bot]: " + resp)

        s = input("[" + name + "]" + ">: ").lower().strip()


if __name__ == '__main__':
    main()
