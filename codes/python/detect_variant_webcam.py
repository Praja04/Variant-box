# -*- coding: utf-8 -*-
"""
Deteksi Variant Kardus / Box Real-Time Menggunakan Webcam & YOLOv8
Model: variant_classifier_v2 (best.pt)
Kelas: Sachet_20gr_LunchBox, Variant_YB77, Variant_BB725
"""

import cv2
import os
import sys
import time
from ultralytics import YOLO

# ==================== KONFIGURASI ====================
# Index webcam (0 = webcam bawaan laptop / USB utama, 1 = webcam eksternal ke-2)
WEBCAM_INDEX = 1    

# Resolusi webcam yang diinginkan
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Ambang batas keyakinan deteksi (Confidence Threshold)
CONF_THRESHOLD = 0.55

# Path Model YOLOv8 Custom
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'pallet', 'runs', 'detect', 
                 'pallet_variant_detection', 'variant_classifier_v2', 'weights', 'best.pt')
)

# Palet warna per kelas (BGR)
CLASS_COLORS = {
    0: (255, 128, 0),    # Cyan/Biru muda - Sachet_20gr_LunchBox
    1: (0, 200, 255),    # Kuning emas - Variant_YB77
    2: (0, 255, 100),    # Hijau neon - Variant_BB725
}

def draw_hud(frame, detected_variants, fps):
    """Menggambar panel informasi (HUD) di bagian atas frame"""
    h, w = frame.shape[:2]
    
    # Header bar semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Garis aksen bawah header
    cv2.line(frame, (0, 55), (w, 55), (0, 200, 255), 2)
    
    # Tampilkan FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 200), 2)
    
    # Tampilkan Status Deteksi
    if detected_variants:
        text_variants = ", ".join(f"{name} ({count})" for name, count in detected_variants.items())
        status_text = f"Terdeteksi: {text_variants}"
        color = (0, 255, 0)
    else:
        status_text = "Mencari Variant Kardus..."
        color = (180, 180, 180)
        
    cv2.putText(frame, status_text, (180, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
    
    # Petunjuk di pojok kanan atas
    cv2.putText(frame, "Tekan 'Q' untuk keluar", (w - 230, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

def main():
    print("=" * 60)
    print("      DETEKSI VARIANT KARDUS WEBCAM (YOLOv8)      ")
    print("=" * 60)

    # 1. Cek & Muat Model YOLO
    model_path = DEFAULT_MODEL
    if not os.path.exists(model_path):
        print(f"[PERINGATAN] Model custom tidak ditemukan di:\n  {model_path}")
        fallback_model = os.path.join(SCRIPT_DIR, "yolov8n.pt")
        if os.path.exists(fallback_model):
            print(f"[INFO] Menggunakan model fallback: {fallback_model}")
            model_path = fallback_model
        else:
            print("[ERROR] File model tidak ditemukan. Pastikan path model benar!")
            return

    print(f"[INFO] Memuat model YOLO: {model_path}")
    try:
        model = YOLO(model_path)
        print("✓ Model YOLO berhasil dimuat!")
        print(f"[INFO] Daftar Kelas Model: {model.names}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model YOLO: {e}")
        return

    # 2. Buka Webcam
    print(f"\n[INFO] Membuka Webcam (Index {WEBCAM_INDEX})...")
    # Gunakan CAP_DSHOW pada Windows untuk start-up webcam yang cepat
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback tanpa CAP_DSHOW
        cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka webcam dengan index {WEBCAM_INDEX}!")
        print("[TIPS] Coba ubah nilai WEBCAM_INDEX menjadi 1 atau pastikan tidak ada aplikasi lain yang menggunakan kamera.")
        return

    # Set resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Webcam siap! Resolusi: {actual_w}x{actual_h}")
    print("\n[PETUNJUK]:")
    print(" - Arahkan kamera ke kardus/kemasan produk.")
    print(" - Tekan tombol 'Q' pada jendela video untuk keluar.\n")

    window_name = "Deteksi Variant Kardus - Webcam (YOLOv8)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[PERINGATAN] Gagal membaca frame dari webcam.")
                time.sleep(0.05)
                continue

            # Hitung FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # 3. Jalankan Prediksi YOLO
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            detected_variants = {}

            # 4. Gambar Bounding Box & Label
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = model.names.get(cls_id, f"Class_{cls_id}")

                    # Catat jumlah variant yang terdeteksi
                    detected_variants[cls_name] = detected_variants.get(cls_name, 0) + 1

                    # Koordinat Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Warna kotak berdasarkan class_id
                    color = CLASS_COLORS.get(cls_id % len(CLASS_COLORS), (0, 255, 0))

                    # Gambar Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Label Teks
                    label_text = f"{cls_name} ({conf*100:.1f}%)"
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Background label tag
                    tag_y1 = max(0, y1 - th - baseline - 8)
                    tag_y2 = y1
                    tag_x2 = min(frame.shape[1], x1 + tw + 10)
                    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
                    
                    # Teks label (warna kontras hitam/putih)
                    cv2.putText(frame, label_text, (x1 + 5, tag_y2 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # 5. Gambar HUD Header
            draw_hud(frame, detected_variants, fps)

            # 6. Tampilkan Frame
            cv2.imshow(window_name, frame)

            # Tombol Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 'q' atau ESC
                print("[INFO] Program dihentikan oleh pengguna.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan via terminal.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam dan jendela video berhasil ditutup. Selesai.")

if __name__ == "__main__":
    main()
