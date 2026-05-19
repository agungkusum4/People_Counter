import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, db
import time

# ================= VARIABLE LOKAL =================
latest_suhu = 0.0  # Variabel lokal untuk menyimpan data suhu terkini

# ================= 1. KONFIGURASI FIREBASE =================
PATH_TO_JSON = "serviceAccountKey.json"
DATABASE_URL = "https://aiot-project-5d7f9-default-rtdb.asia-southeast1.firebasedatabase.app/"

print("Menginisialisasi Firebase...")
try:
    cred = credentials.Certificate(PATH_TO_JSON)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })
    print("Firebase Berhasil Terhubung!")
except Exception as e:
    print(f"Gagal koneksi Firebase: {e}")
    exit()

# Reference ke node '/sensor' di Firebase Realtime Database
firebase_ref = db.reference('/sensor')

# ================= 2. KONFIGURASI MQTT =================
# Jika pakai Mosquitto lokal di laptop, gunakan "localhost" atau IP Laptop/Raspi
# Jika mau tes pakai internet dulu tanpa instal broker, bisa ganti ke "broker.hivemq.com"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_SUHU = "ruangan/sensor/suhu"

# Fungsi saat Python sukses konek ke MQTT Broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Terhubung ke MQTT Broker!")
        client.subscribe(MQTT_TOPIC_SUHU)
        print(f"Mendengarkan topik: {MQTT_TOPIC_SUHU}")
    else:
        print(f"Gagal konek ke MQTT, return code: {rc}")

# Fungsi saat ada data suhu masuk dari ESP32
def on_message(client, userdata, msg):
    global latest_suhu
    try:
        # 1. Ambil data dari MQTT dan simpan ke variabel lokal
        payload_string = msg.payload.decode("utf-8")
        latest_suhu = float(payload_string)
        print(f"\n[MQTT] Masuk data suhu baru: {latest_suhu} °C (Tersimpan di variabel lokal)")
        
    except Exception as e:
        print(f"Error membaca pesan MQTT: {e}")

# Inisialisasi MQTT Client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Menghubungkan ke MQTT Broker...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e:
    print(f"Gagal terhubung ke Broker: {e}")
    exit()

# Jalankan MQTT di background agar loop utama tidak terganggu
client.loop_start()

# ================= 3. LOOP UTAMA (KIRIM KE FIREBASE) =================
print("Sistem Testing Suhu Siap. Menunggu data dari ESP32...")
try:
    while True:
        # Cek apakah variabel lokal sudah terisi (bukan 0.0)
        if latest_suhu != 0.0:
            print(f"[Loop Utama] Mengambil nilai dari variabel lokal: {latest_suhu} °C")
            print("[Firebase] Mengirim ke Realtime Database...")
            
            # Kirim data variabel lokal tersebut ke Firebase
            firebase_ref.update({
                'suhu': latest_suhu
            })
            print("[Firebase] Update Berhasil!\n")
        else:
            print("[Sistem] Menunggu kiriman suhu pertama dari ESP32...")
            
        # Jeda pengiriman ke Firebase (misal dievaluasi setiap 10 detik)
        time.sleep(30)

except KeyboardInterrupt:
    print("\nSistem dihentikan.")
    client.loop_stop()