import socket
import cv2
import pickle
import struct
import threading
import pyshine as ps
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def process_frame(frame):
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)

        cv2.putText(image, str(shoulder_elbow_angle),
                    tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if counter != 0:
            if shoulder_elbow_angle < 80:
                cv2.putText(image, 'Too Low, Keep your arms InLine to your shoulder',
                            tuple(np.multiply([0.6, 0.4], [frame.shape[1], frame.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        if shoulder_elbow_angle < 80:
            stage = "Down"
        if shoulder_elbow_angle > 160 and stage == 'Down':
            stage = "Up"
            counter += 1

    except:
        pass

    cv2.rectangle(image, (0, 0), (150, 73), (245, 117, 16), -1)
    cv2.putText(image, "REPS", (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 63, 125), 2, cv2.LINE_AA)

    cv2.putText(image, "STAGE", (75, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if stage == 'Up':
        cv2.putText(image, stage, (78, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 63, 125), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, stage, (60, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 63, 125), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    return image


def show_client(addr, client_socket):
    try:
        print('CLIENT {} CONNECTED!'.format(addr))
        if client_socket:
            data = b""
            payload_size = struct.calcsize("Q")
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)
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
                frame = process_frame(frame)
                cv2.imshow(f"FROM {addr}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            client_socket.close()
    except Exception as e:
        print(f"CLIENT {addr} DISCONNECTED")
        pass


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = "192.168.198.208"
print('HOST IP:', host_ip)
port = 9999
socket_address = (host_ip, port)

server_socket.bind(socket_address)
server_socket.listen()
print("LISTENING AT:", socket_address)

pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)
counter = 0
stage = None

while True:
    client_socket, addr = server_socket.accept()
    thread = threading.Thread(target=show_client, args=(addr, client_socket))
    thread.start()
    print("TOTAL CLIENTS ", threading.activeCount() - 1)
