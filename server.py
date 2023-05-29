import socket
import cv2
import pickle
import struct
import threading
import pyshine as ps
import cv2
import imutils
import numpy as np
import mediapipe as mp

# Load the pre-trained pose detection model
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = "192.168.198.208" # put your host IP address here
print('HOST IP:', host_ip)
port = 9999
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at", socket_address)


def calculate_angle(a, b, c):
    """
    Calculates the angle between three points in a 2D plane.
    :param a: A tuple or list containing the (x,y) coordinates of the first point.
    :type a: tuple or list of int or float
    :param b: A tuple or list containing the (x,y) coordinates of the second point.
    :type b: tuple or list of int or float
    :param c: A tuple or list containing the (x,y) coordinates of the third point.
    :type c: tuple or list of int or float
    :return: The angle in degrees between the line segments connecting point a to b and point b to c.
    :rtype: float
    """
    a = np.array(a)  # First
    b = np.array(b)  # Second
    c = np.array(c)  # Third

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def detect_shoulder_angle(frame):
    # Recolour the frame to Mediapipe format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make pose detections
    results = pose_detector.process(image)

    # Recolour the frame to OpenCV format
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract shoulder, elbow, and wrist landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate shoulder-elbow angle
        shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Visualize the angle
        cv2.putText(image, str(shoulder_elbow_angle),
                    tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    except:
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
                frame = detect_shoulder_angle(frame)  # Perform shoulder form tracking on the frame
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
