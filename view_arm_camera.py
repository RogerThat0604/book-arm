import cv2

cap = cv2.VideoCapture(6)

ret, frame = cap.read()

if not ret:
    print("프레임 실패")
else:
    h, w = frame.shape[:2]

    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv2.line(frame, (0, h//2), (w, h//2), (255, 255, 255), 2)

    cv2.imwrite("/home/jetcobot/arm_camera_test.jpg", frame)
    print("저장 완료: /home/jetcobot/arm_camera_test.jpg")

cap.release()