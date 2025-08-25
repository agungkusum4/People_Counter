import cv2
import numpy as np
from collections import OrderedDict
import paho.mqtt.client as mqtt
import json

# ------------------------ CENTROID TRACKER ------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# ------------------------ MQTT SETUP ------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "peoplecounter/data"

client = mqtt.Client(protocol=mqtt.MQTTv311)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ------------------------ LOAD MODEL ------------------------
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ------------------------ VIDEO INPUT ------------------------
cap = cv2.VideoCapture(0)

left_count = 0
right_count = 0

ct = CentroidTracker()
previous_x = {}

line_x = None

# ------------------------ MAIN LOOP ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    if line_x is None:
        line_x = w // 2  # garis vertikal tengah frame

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    centroids = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    objects = ct.update(np.array(centroids))

    for (object_id, centroid) in objects.items():
        current_x = centroid[0]

        if object_id in previous_x:
            prev_x = previous_x[object_id]

            if prev_x > line_x and current_x < line_x:
                left_count += 1
                print(f"Person {object_id} moved LEFT.")

                # Kirim ke MQTT saat bergerak ke kiri
                payload = f"Masuk: {left_count}, Keluar: {right_count}"
                client.publish(MQTT_TOPIC, payload)


            elif prev_x < line_x and current_x > line_x:
                right_count += 1
                print(f"Person {object_id} moved RIGHT.")

                # Kirim ke MQTT saat bergerak ke kanan
                payload = f"Masuk: {left_count}, Keluar: {right_count}"
                client.publish(MQTT_TOPIC, payload)

        previous_x[object_id] = current_x

    # Gambar garis tengah dan info
    cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
    cv2.putText(frame, f"LEFT (Masuk): {left_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"RIGHT (Keluar): {right_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counting with MQTT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------ CLEANUP ------------------------
cap.release()
cv2.destroyAllWindows()
client.disconnect()
print("Program selesai. MQTT client sudah disconnect.")