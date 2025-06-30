# app.py
import cv2
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO
from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import torch
import math

# --- 1. Konfigurasi Model Deteksi Objek (YOLOv8) ---
try:
    model = YOLO('yolov8s.pt')
    print("Model YOLOv8 berhasil dimuat.")
    if torch.cuda.is_available():
        model.to('cuda')
        print("Model berhasil dipindahkan ke GPU (CUDA).")
    else:
        print("GPU (CUDA) tidak tersedia. Model akan berjalan di CPU.")
except Exception as e:
    print(f"Error saat memuat model YOLOv8: {e}")
    print("Pastikan Anda memiliki koneksi internet untuk mengunduh model saat pertama kali dijalankan.")
    exit()

# --- 2. Konfigurasi Deteksi dan Penghitungan ---
# URL stream dari data yang Anda berikan
video_source = "https://stream.kuduskab.go.id/memfs/e997882d-1a7d-43a6-ae87-93b07e469777.m3u8"
vehicle_classes_yolo = ["car", "motorcycle", "bus", "truck"]
confidence_threshold = 0.4
colors = {name: [int(c) for c in np.random.uniform(0, 255, 3)] for name in model.names}

# Mapping kelas YOLO ke kategori dashboard
class_mapping = {
    'motorcycle': 'Sepeda Motor',
    'car': 'Mobil Penumpang',
    'bus': 'Bus Besar',
    'truck': 'Truk Barang',
    # Tambahkan kategori 'Kendaraan Sedang' jika model mendeteksinya
}

# Inisialisasi Flask App
app = Flask(__name__)
CORS(app)

# --- LOGIKA PENGHITUNGAN KENDARAAN DUA ARAH ---
# Definisikan dua garis penghitungan (arah Normal dan Opposite)
# Sesuaikan koordinat Y ini dengan video stream Anda!
# Contoh: Garis di bagian atas dan bawah frame
line_y_normal = 450
line_y_opposite = 550
line_x_start = 0
line_x_end = 1280

# Garis-garis untuk visualisasi
line_normal = [(line_x_start, line_y_normal), (line_x_end, line_y_normal)]
line_opposite = [(line_x_start, line_y_opposite), (line_x_end, line_y_opposite)]

# Variabel untuk melacak kendaraan
# {id: {'class': 'car', 'centroid': (x, y), 'direction': None, 'counted': False}}
tracked_vehicles = {}
vehicle_id_counter = 0

# Data untuk grafik dashboard (per 5 menit dan harian)
current_counts_5_min = Counter({'Normal': Counter(), 'Opposite': Counter()})
daily_counts_hourly = {} # {hour: {category: count}}

def get_centroid(box):
    """Menghitung titik tengah bounding box."""
    x1, y1, x2, y2 = [int(c) for c in box]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def update_tracker_and_count(detections):
    """Update tracker dengan deteksi baru dan hitung kendaraan."""
    global tracked_vehicles, vehicle_id_counter, current_counts_5_min, daily_counts_hourly
    
    current_centroids = {}
    for box in detections.boxes:
        label = model.names[int(box.cls[0])]
        if label in vehicle_classes_yolo:
            centroid = get_centroid(box.xyxy[0])
            current_centroids[centroid] = {'label': label, 'box': box.xyxy[0]}

    # Update tracker
    updated_ids = set()
    for vid, data in list(tracked_vehicles.items()):
        min_dist = float('inf')
        closest_centroid = None
        for centroid_curr, obj_data in current_centroids.items():
            dist = math.hypot(data['centroid'][0] - centroid_curr[0], data['centroid'][1] - centroid_curr[1])
            if dist < min_dist:
                min_dist = dist
                closest_centroid = centroid_curr

        if closest_centroid and min_dist < 100:
            # Update posisi centroid
            tracked_vehicles[vid]['centroid_prev'] = tracked_vehicles[vid]['centroid']
            tracked_vehicles[vid]['centroid'] = closest_centroid
            updated_ids.add(vid)
        else:
            # Hapus kendaraan yang hilang dari frame
            del tracked_vehicles[vid]

    # Tambahkan deteksi baru
    for centroid_curr, obj_data in current_centroids.items():
        is_new = True
        for vid in updated_ids:
            if math.hypot(tracked_vehicles[vid]['centroid'][0] - centroid_curr[0], tracked_vehicles[vid]['centroid'][1] - centroid_curr[1]) < 100:
                is_new = False
                break
        if is_new:
            tracked_vehicles[vehicle_id_counter] = {
                'class': obj_data['label'],
                'centroid': centroid_curr,
                'centroid_prev': centroid_curr, # Inisialisasi centroid_prev
                'direction': None,
                'counted_normal': False,
                'counted_opposite': False
            }
            vehicle_id_counter += 1

    # Cek dan hitung kendaraan yang melewati garis
    for vid, data in tracked_vehicles.items():
        centroid_prev = data['centroid_prev']
        centroid_curr = data['centroid']
        
        # Cek arah Normal (dari bawah ke atas)
        if not data['counted_normal'] and centroid_prev[1] > line_y_normal and centroid_curr[1] <= line_y_normal:
            data['counted_normal'] = True
            category = class_mapping.get(data['class'], 'Lain-lain')
            current_counts_5_min['Normal'][category] += 1
            print(f"DEBUG: {category} melintasi garis Normal.")
            
            # Agregasi data harian
            current_hour = time.strftime("%H", time.localtime())
            if current_hour not in daily_counts_hourly:
                daily_counts_hourly[current_hour] = Counter()
            daily_counts_hourly[current_hour][category] += 1

        # Cek arah Opposite (dari atas ke bawah)
        if not data['counted_opposite'] and centroid_prev[1] < line_y_opposite and centroid_curr[1] >= line_y_opposite:
            data['counted_opposite'] = True
            category = class_mapping.get(data['class'], 'Lain-lain')
            current_counts_5_min['Opposite'][category] += 1
            print(f"DEBUG: {category} melintasi garis Opposite.")

def generate_frames():
    """Generator function to stream processed frames."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka sumber video.")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        # Lakukan deteksi objek
        results = model(frame, conf=confidence_threshold, verbose=False)
        for result in results:
            update_tracker_and_count(result)

        # Gambar garis penghitungan dan label
        cv2.line(frame, line_normal[0], line_normal[1], (0, 255, 0), 2)
        cv2.putText(frame, "Normal", (line_normal[0][0] + 10, line_y_normal - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.line(frame, line_opposite[0], line_opposite[1], (255, 0, 0), 2)
        cv2.putText(frame, "Opposite", (line_opposite[0][0] + 10, line_y_opposite - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Gambar bounding box dan label
        display_text = "Detected:"
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                label = model.names[int(box.cls[0])]
                if label in vehicle_classes_yolo:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    color = colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hitung dan tampilkan FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 3. Rute Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """Rute API untuk mengirim data hitungan kendaraan untuk grafik."""
    global current_counts_5_min, daily_counts_hourly
    
    # Format data untuk dikirim ke frontend
    total_normal = sum(current_counts_5_min['Normal'].values())
    total_opposite = sum(current_counts_5_min['Opposite'].values())
    total_all = total_normal + total_opposite
    
    # Data volume harian (contoh data dummy jika belum ada)
    # Anda bisa mengisi daily_counts_hourly dari database di sini
    if not daily_counts_hourly:
         # Contoh data untuk hari ini, diupdate setiap jam
         # Ini akan diganti dengan data real-time dari loop
         daily_counts_hourly = {
            '08': {'Sepeda Motor': 150, 'Mobil Penumpang': 70},
            '09': {'Sepeda Motor': 200, 'Mobil Penumpang': 100},
            '10': {'Sepeda Motor': 350, 'Mobil Penumpang': 150},
            '11': {'Sepeda Motor': 400, 'Mobil Penumpang': 200},
         }

    response_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'volume_per_category_normal': current_counts_5_min['Normal'],
        'volume_per_category_opposite': current_counts_5_min['Opposite'],
        'total_volume': total_all,
        'daily_volume_hourly': daily_counts_hourly
    }
    
    return jsonify(response_data)

# --- 4. Jalankan Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)