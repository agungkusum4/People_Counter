import cv2
import numpy as np
from collections import OrderedDict
# --- Tambahkan library Firebase ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# --- Inisialisasi Firebase SDK ---
cred = credentials.Certificate('serviceAccountKey.json') # Pastikan nama file json sesuai
firebase_admin.initialize_app(cred, {
    # PASTIKAN URL database_url ini sama persis dengan yang ada di config Firebase kamu
    'databaseURL': 'https://aiot-project-5d7f9-default-rtdb.asia-southeast1.firebasedatabase.app/' 
})

# Data masuk ke folder sensor
firebase_ref = db.reference('sensor')

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

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)

left_count = 0 #masuk
right_count = 0 #keluar
total_in_room = 0 #total orang

ct = CentroidTracker()
previous_x = {}
line_x = None

# Kirim data awal (0 orang) saat program pertama kali dijalankan
try:
    firebase_ref.update({'jumlah_orang': total_in_room})
except Exception as e:
    print("Gagal inisialisasi ke Firebase:", e)

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

    objects = ct.update(centroids)

    current_object_ids = list(objects.keys())
    for obj_id in list(previous_x.keys()):
        if obj_id not in current_object_ids:
            del previous_x[obj_id]

    # Variabel bendera (flag) untuk mengetahui apakah ada perubahan jumlah orang di loop ini
    data_berubah = False

    for (object_id, centroid) in objects.items():
        current_x = centroid[0]

        if object_id in previous_x:
            prev_x = previous_x[object_id]

            # Jika bergerak dari kanan ke kiri (Masuk)
            if prev_x > line_x and current_x < line_x:
                left_count += 1
                total_in_room += 1  
                data_berubah = True # Ada perubahan data
                print(f"Person {object_id} moved LEFT (Entered). Total in room: {total_in_room}")

            # Jika bergerak dari kiri ke kanan (Keluar)
            elif prev_x < line_x and current_x > line_x:
                right_count += 1
                total_in_room = max(0, total_in_room - 1)  
                data_berubah = True # Ada perubahan data
                print(f"Person {object_id} moved RIGHT (Exited). Total in room: {total_in_room}")

        previous_x[object_id] = current_x

    # --- JIKA ADA PERUBAHAN, KIRIM KE FIREBASE ---
    if data_berubah:
        try:
            # Menggunakan .update() agar data suhu/daya yang dikirim ESP32 tidak terhapus
            firebase_ref.update({'jumlah_orang': total_in_room})
            print(">>> Data jumlah orang berhasil diupdate ke Firebase!")
        except Exception as e:
            print(">>> Gagal mengirim data ke Firebase:", e)

    # Gambar garis vertikal di tengah frame
    cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)

    # Tampilkan hasil counting di layar kamera
    cv2.putText(frame, f"LEFT (In): {left_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"RIGHT (Out): {right_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(frame, f"TOTAL IN ROOM: {total_in_room}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    cv2.imshow("People Counting - Vertical Line", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()