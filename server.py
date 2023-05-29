import socket
import cv2
import pickle
import struct
import threading
import pyshine as ps
import cv2
import imutils

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = "192.168.198.208" # put your host IP address here
print('HOST IP:', host_ip)
port = 9999
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at", socket_address)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


def show_client(addr, client_socket):
    try:
        print('CLIENT {} CONNECTED!'.format(addr))
        if client_socket:  # if a client socket exists
            data = b""
            payload_size = struct.calcsize("Q")
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)  # 4K
                    if not packet:
                        break
                    data += packet
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(4 * 1024)
                frame_data = data[:msg_size]
                data = data[msg_size:]
                frame = pickle.loads(frame_data)
                text = f"CLIENT: {addr}"
                frame = ps.putBText(frame, text, 10, 10, vspace=10, hspace=1, font_scale=0.7,
                                    background_RGB=(255, 0, 0), text_RGB=(255, 250, 250))
                frame = detect_faces(frame)  # Perform face detection on the frame
                cv2.imshow(f"FROM {addr}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            client_socket.close()
    except Exception as e:
        print(f"CLIENT {addr} DISCONNECTED")
        pass


while True:
    client_socket, addr = server_socket.accept()
    thread = threading.Thread(target=show_client, args=(addr, client_socket))
    thread.start()
    print("TOTAL CLIENTS ", threading.activeCount() - 1)
