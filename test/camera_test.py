import cv2

# 캠 열기
cap = cv2.VideoCapture('/dev/jetcocam0')

# 캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
 
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break
 
    # 화면에 출력
    cv2.imshow('Cam', frame)
 
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
