import socket
import cv2
import pickle
import struct
import threading
import cv2
import imutils
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify

app = Flask(__name__)

# Load the pre-trained pose detection model
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = "192.168.0.137"  # put your host IP address here
print('HOST IP:', host_ip)
port = 9999
socket_address = (host_ip, port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at", socket_address)


def calculate_angle(a, b, c):
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

        if shoulder_elbow_angle < 90:
            print("angle less than 90")

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
                    packet = client_socket.recv(1024)  # 4K
                    if not packet:
                        break
                    data += packet
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(1024)
                frame_data = data[:msg_size]
                data = data[msg_size:]
                frame = pickle.loads(frame_data)

                frame = detect_shoulder_angle(frame)  # Perform shoulder form tracking on the frame

                # cv2.imshow(f"FROM {addr}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            client_socket.close()
    except Exception as e:
        print(f"CLIENT {addr} DISCONNECTED")
        pass


@app.route("/shoulder-angle", methods=["GET"])
def get_shoulder_angle():
    # Perform shoulder form tracking on a sample frame
    sample_frame = cv2.imread("sample_frame.jpg")  # Replace "sample_frame.jpg" with your actual frame

    angle = detect_shoulder_angle(sample_frame)
    if angle < 90:
        return jsonify({"message": "Shoulder angle is less than 90 degrees."})
    else:
        return jsonify({"message": "Shoulder angle is greater than or equal to 90 degrees."})


if __name__ == "__main__":
    thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000, "threaded": True})
    thread.start()
    while True:
        client_socket, addr = server_socket.accept()
        client_thread = threading.Thread(target=show_client, args=(addr, client_socket))
        client_thread.start()
        print("TOTAL CLIENTS ", threading.activeCount() - 2)
