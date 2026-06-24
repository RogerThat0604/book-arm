import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"camera index {i} 열림")
        ret, frame = cap.read()
        print("frame:", ret, None if frame is None else frame.shape)
        cap.release()
