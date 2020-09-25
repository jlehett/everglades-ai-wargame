import socket
import sys
import time

while True:
    # Delay time between socket connections
    time.sleep(.2)

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 10000)
    print('connecting to %s port %s' % server_address, file=sys.stderr)
    sock.connect(server_address)

    try:
        # Send data
        pinput = input("Type a message to send the server: ")
        # Convert input string into binary to send to server
        #message = b''.join(format(ord(i), 'b') for i in pinput) #b'This is the message.  It will be repeated.'

        print('sending %s' % pinput)
        sock.sendall(pinput.encode('utf-8'))

        # Look for the response
        amount_received = 0
        amount_expected = len(pinput)
        
        while amount_received < amount_expected:
            data = sock.recv(16)
            amount_received += len(data)
            print('received "%s"' % data, file=sys.stderr)

    finally:
        print("closing socket", file=sys.stderr)
        sock.close()
