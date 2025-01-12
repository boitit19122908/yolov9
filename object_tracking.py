import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config value
video_path = "/home/boiinlove/capstone project of Kiet senpai/yolov9/data_ext/people.avi"
output_video_path = "/home/boiinlove/capstone project of Kiet senpai/yolov9/data_ext/output.avi"
conf_threshold = 0.5
tracking_class = None  # None: track all

# Khởi tạo DeepSort (không có tham số min_confidence)
tracker = DeepSort(max_age=30, nms_max_overlap=0.5)

# Khởi tạo YOLOv9
device = "cpu"  # "cuda": GPU, "cpu": CPU, "mps:0"
model = DetectMultiBackend(weights="/home/boiinlove/capstone project of Kiet senpai/yolov9/weights/yolov9-c.pt", device=device, fuse=True)

model = AutoShape(model)

# Load classname từ file classes.names
with open("/home/boiinlove/capstone project of Kiet senpai/yolov9/data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

# Lấy thông tin từ video gốc
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Khởi tạo VideoWriter để ghi video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Tiến hành đọc từng frame từ video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Processing completed.")
        break  # Dừng nếu không đọc được frame

    # Đưa qua model để detect
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        # Lọc theo conf_threshold và class_id
        if (tracking_class is None or class_id == tracking_class) and confidence >= conf_threshold:
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Cập nhật, gán ID bằng DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # Duyệt qua các đối tượng đang được theo dõi
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ và class_id từ track
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()

            # Lấy độ tin cậy (confidence) từ đối tượng detect ban đầu
            conf = next((conf for bbox, conf, cls in detect if cls == class_id), 0)

            # Kiểm tra nếu độ tin cậy cao hơn ngưỡng
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            # Tạo nhãn với tên lớp và ID của đối tượng
            label = "{}-{} ({:.2f})".format(class_names[class_id], track_id, conf)

            # Vẽ khung chữ nhật quanh đối tượng
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)

            # Vẽ nền cho nhãn
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)

            # Vẽ nhãn lên hình ảnh
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Ghi frame đã xử lý vào video output
    out.write(frame)

    # Hiển thị frame lên màn hình
    cv2.imshow("Object Tracking", frame)

    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
out.release()
cv2.destroyAllWindows()
