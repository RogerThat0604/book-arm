import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(i, frame.shape)
        cv2.imwrite(f"/home/jetcobot/camera_{i}.jpg", frame)
    cap.release()
