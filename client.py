import socket,cv2, pickle,struct
import imutils # pip install imutils
camera = True
if camera == True:
	vid = cv2.VideoCapture(0)
else:
	vid = cv2.VideoCapture('videos/mario.mp4')
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.0.137' # Here according to your server ip write the address

i = 9999
port = i
client_socket.connect((host_ip,port))

if client_socket:
	while (vid.isOpened()):
		try:
			img, frame = vid.read()
			frame = cv2.flip(frame, 1)
			frame = imutils.resize(frame,width=380)
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			cv2.imshow(f"TO: {host_ip}",frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				client_socket.close()
		except:
			print('VIDEO FINISHED!')
			break

