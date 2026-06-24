import cv2
import socket
import struct
import time

PC_IP = "192.168.0.52"
PC_PORT = 5000

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

JPEG_QUALITY = 60
MAX_PACKET_SIZE = 1400

qr_detector = cv2.QRCodeDetector()


def send_udp_frame(sock, addr, frame, frame_id):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not ok:
        return

    data = encoded.tobytes()

    total_packets = (
        len(data) + MAX_PACKET_SIZE - 1
    ) // MAX_PACKET_SIZE

    for packet_id in range(total_packets):
        start = packet_id * MAX_PACKET_SIZE
        end = start + MAX_PACKET_SIZE

        chunk = data[start:end]

        header = struct.pack(
            "!IHH",
            frame_id,
            packet_id,
            total_packets
        )

        sock.sendto(
            header + chunk,
            addr
        )


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

addr = (
    PC_IP,
    PC_PORT
)

frame_id = 0

print("QR Detection Test Start")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        data, points, _ = qr_detector.detectAndDecode(frame)

        h, w = frame.shape[:2]

        center_x = w // 2

        cv2.line(
            frame,
            (center_x, 0),
            (center_x, h),
            (255, 0, 0),
            2
        )

        if points is not None and data:

            pts = points[0].astype(int)

            for i in range(4):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i + 1) % 4])

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (0, 255, 0),
                    2
                )

            cx = int(sum(p[0] for p in pts) / 4)
            cy = int(sum(p[1] for p in pts) / 4)

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            error_x = cx - center_x

            cv2.putText(
                frame,
                f"QR={data}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"error_x={error_x}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            print(
                f"QR={data}, "
                f"cx={cx}, "
                f"error_x={error_x}"
            )

        else:

            cv2.putText(
                frame,
                "NO QR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        send_udp_frame(
            sock,
            addr,
            frame,
            frame_id
        )

        frame_id += 1

        time.sleep(0.03)

except KeyboardInterrupt:
    print("종료")

finally:
    cap.release()
    sock.close()