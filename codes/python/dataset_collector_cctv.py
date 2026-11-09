# -*- coding: utf-8 -*-
"""
Dataset Collector dari CCTV Palletizing
Fitur:
  - ROI selection (gambar kotak area pallet)
  - Perspective correction (klik 4 titik sudut pallet -> koreksi kemiringan)
  - Auto-capture tiap N detik
  - Simpan ke folder siap labeling di Roboflow / CVAT
By: Yaser Ali Husen / Antigravity AI
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime

# ==================== THREADED VIDEO CAPTURE ====================
class ThreadedVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, str) and ("rtsp" in src or "http" in src):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret, self.frame = ret, frame
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

# ==================== CONFIGURATION ====================
VIDEO_SOURCE = "rtsp://admin:1q2w3e4r@10.11.8.137:554/Streaming/Channels/101"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
OUTPUT_SIZE = (640, 640)   # Ukuran output gambar dataset (kotak, ideal untuk YOLO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset_cctv'))
os.makedirs(DATASET_DIR, exist_ok=True)

CAPTURE_INTERVAL_SEC = 3   # Auto-capture tiap N detik

# ==================== PERSPECTIVE CORRECTION ====================
class PerspectiveSelector:
    """
    User mengklik 4 titik sudut pallet pada frame CCTV.
    Sistem akan melakukan perspective warp agar pallet tampak lurus (bird's eye view).
    """
    def __init__(self):
        self.points = []
        self.temp_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def select_corners(self, frame):
        """
        Minta user klik 4 sudut pallet secara berurutan:
        1=Kiri-Atas, 2=Kanan-Atas, 3=Kanan-Bawah, 4=Kiri-Bawah

        Return: (transform_matrix, dst_size) atau (None, None) jika di-skip
        """
        self.points = []
        self.temp_frame = frame.copy()
        win_name = 'KLIK 4 SUDUT PALLET (Kiri-Atas -> Kanan-Atas -> Kanan-Bawah -> Kiri-Bawah)'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        corner_labels = [
            "1. Klik KIRI-ATAS pallet",
            "2. Klik KANAN-ATAS pallet",
            "3. Klik KANAN-BAWAH pallet",
            "4. Klik KIRI-BAWAH pallet"
        ]

        print("\n" + "=" * 60)
        print("KOREKSI KEMIRINGAN (PERSPECTIVE CORRECTION)")
        print("=" * 60)
        print("  Klik 4 sudut pallet secara berurutan:")
        print("  1. Kiri-Atas  ->  2. Kanan-Atas")
        print("  3. Kanan-Bawah  ->  4. Kiri-Bawah")
        print("  Tekan ENTER setelah 4 titik, ESC untuk skip.")
        print("=" * 60)

        while True:
            display = self.temp_frame.copy()

            # Instruksi bar atas
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            if len(self.points) < 4:
                instruction = corner_labels[len(self.points)]
            else:
                instruction = "4 titik sudah dipilih! Tekan ENTER untuk konfirmasi."

            cv2.putText(display, instruction,
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "ENTER = OK | ESC = Skip (tanpa koreksi) | R = Reset titik",
                        (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (180, 180, 180), 1, cv2.LINE_AA)

            # Gambar titik-titik yang sudah di-klik
            colors = [(0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255)]
            labels = ["KA", "KaA", "KaB", "KB"]

            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 8, colors[i], -1)
                cv2.circle(display, pt, 10, (255, 255, 255), 2)
                cv2.putText(display, labels[i], (pt[0] + 12, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            # Gambar garis antar titik
            if len(self.points) >= 2:
                for i in range(len(self.points)):
                    j = (i + 1) % len(self.points)
                    if j < len(self.points):
                        cv2.line(display, self.points[i], self.points[j],
                                 (0, 255, 0), 2, cv2.LINE_AA)
                # Tutup polygon jika 4 titik
                if len(self.points) == 4:
                    cv2.line(display, self.points[3], self.points[0],
                             (0, 255, 0), 2, cv2.LINE_AA)

                    # Preview area yang akan di-warp (semi-transparan)
                    pts = np.array(self.points, dtype=np.int32)
                    mask_overlay = display.copy()
                    cv2.fillPoly(mask_overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(mask_overlay, 0.15, display, 0.85, 0, display)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13 and len(self.points) == 4:  # ENTER
                cv2.destroyWindow(win_name)
                # Hitung perspective transform matrix
                src_pts = np.float32(self.points)
                # Destination: kotak lurus
                w_out, h_out = OUTPUT_SIZE
                dst_pts = np.float32([
                    [0, 0],
                    [w_out - 1, 0],
                    [w_out - 1, h_out - 1],
                    [0, h_out - 1]
                ])
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                print(f"[OK] Perspective transform siap! 4 titik: {self.points}")
                return matrix, OUTPUT_SIZE

            elif key == 27:  # ESC
                cv2.destroyWindow(win_name)
                print("[INFO] Perspective correction di-skip.")
                return None, None

            elif key in (ord('r'), ord('R')):
                self.points = []
                print("[INFO] Titik di-reset. Klik ulang 4 sudut.")


# ==================== ROI SELECTOR (SIMPLE RECTANGLE) ====================
class ROISelector:
    def __init__(self):
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.roi = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_pt = (x, y)
            x1 = min(self.start_pt[0], self.end_pt[0])
            y1 = min(self.start_pt[1], self.end_pt[1])
            x2 = max(self.start_pt[0], self.end_pt[0])
            y2 = max(self.start_pt[1], self.end_pt[1])
            if (x2 - x1) > 50 and (y2 - y1) > 50:
                self.roi = (x1, y1, x2, y2)

    def select(self, frame):
        self.roi = None
        win_name = 'GAMBAR KOTAK AREA PALLET (ENTER = OK, ESC = Skip)'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        while True:
            display = frame.copy()
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            cv2.putText(display,
                "GAMBAR KOTAK di area pallet | ENTER = OK | ESC = Skip",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if self.drawing and self.start_pt and self.end_pt:
                cv2.rectangle(display, self.start_pt, self.end_pt, (0, 255, 0), 2)

            if self.roi:
                x1, y1, x2, y2 = self.roi
                dimmed = cv2.addWeighted(display, 0.3, np.zeros_like(display), 0.7, 0)
                dimmed[y1:y2, x1:x2] = display[y1:y2, x1:x2]
                display = dimmed
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13 and self.roi:
                cv2.destroyWindow(win_name)
                return self.roi
            elif key == 27:
                cv2.destroyWindow(win_name)
                return None


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("DATASET COLLECTOR - CCTV PALLETIZING")
    print("Untuk Training Model YOLO Baru (Sudut CCTV)")
    print("=" * 60)

    # 1. Connect
    print(f"[INFO] Menghubungkan ke: {VIDEO_SOURCE}...")
    cap = ThreadedVideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Gagal terhubung ke CCTV!")
        return
    print("[OK] CCTV terhubung!")

    time.sleep(1)
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame!")
        cap.release()
        return
    first_frame = cv2.resize(first_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # 2. Pilih mode: ROI kotak ATAU 4-titik perspective
    print("\n  Pilih mode crop:")
    print("  [1] ROI Kotak (gambar persegi) + Zoom")
    print("  [2] Perspective Correction (klik 4 sudut pallet)")
    print("  [3] Full frame (tanpa crop)")

    mode = None
    active_roi = None
    perspective_matrix = None
    perspective_size = None

    win_choose = "PILIH MODE: Tekan 1, 2, atau 3"
    cv2.namedWindow(win_choose, cv2.WINDOW_NORMAL)
    while True:
        display = first_frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        cv2.putText(display, "PILIH MODE CROP:",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, "[1] ROI Kotak + Zoom   [2] Perspective 4-Titik   [3] Full Frame",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win_choose, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('1'):
            mode = 'roi'
            break
        elif key == ord('2'):
            mode = 'perspective'
            break
        elif key == ord('3'):
            mode = 'full'
            break
    cv2.destroyWindow(win_choose)

    # 3. Setup mode
    if mode == 'roi':
        roi_sel = ROISelector()
        active_roi = roi_sel.select(first_frame)
        if active_roi:
            print(f"[OK] ROI: {active_roi}")
        else:
            mode = 'full'

    elif mode == 'perspective':
        persp_sel = PerspectiveSelector()
        perspective_matrix, perspective_size = persp_sel.select_corners(first_frame)
        if perspective_matrix is None:
            mode = 'full'

    print(f"\n[INFO] Mode aktif: {mode}")
    print(f"[INFO] Output dataset: {DATASET_DIR}")
    print(f"[INFO] Auto-capture tiap {CAPTURE_INTERVAL_SEC} detik")
    print("\n[PETUNJUK]")
    print("  SPACE = Manual capture")
    print("  A     = Toggle auto-capture ON/OFF")
    print("  Q     = Quit")
    print("=" * 60)

    cv2.namedWindow('Dataset Collector', cv2.WINDOW_NORMAL)

    total_captured = 0
    auto_capture = False
    last_auto_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Apply transform sesuai mode
        if mode == 'roi' and active_roi:
            rx1, ry1, rx2, ry2 = active_roi
            crop = frame[ry1:ry2, rx1:rx2]
            work_frame = cv2.resize(crop, OUTPUT_SIZE)
        elif mode == 'perspective' and perspective_matrix is not None:
            work_frame = cv2.warpPerspective(
                frame, perspective_matrix, perspective_size)
        else:
            work_frame = cv2.resize(frame, OUTPUT_SIZE)

        # Display frame
        display = work_frame.copy()
        dh, dw = display.shape[:2]

        # Info panel atas
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (dw, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        auto_text = "AUTO: ON" if auto_capture else "AUTO: OFF"
        auto_color = (0, 255, 0) if auto_capture else (100, 100, 100)
        cv2.putText(display, f"Dataset Collector | Mode: {mode.upper()} | {auto_text}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(display, f"Captured: {total_captured} | Output: {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Auto-capture status indicator
        cv2.circle(display, (dw - 20, 15), 8, auto_color, -1)

        # Panel bawah
        cv2.rectangle(display, (0, dh - 25), (dw, dh), (25, 25, 25), -1)
        cv2.putText(display, "[SPACE] Capture  [A] Toggle Auto  [Q] Quit",
                    (10, dh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow('Dataset Collector', display)

        # Auto-capture timer
        should_capture = False
        if auto_capture and (time.time() - last_auto_time) >= CAPTURE_INTERVAL_SEC:
            should_capture = True
            last_auto_time = time.time()

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            should_capture = True

        if should_capture:
            total_captured += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"pallet_cctv_{ts}.jpg"
            filepath = os.path.join(DATASET_DIR, filename)
            cv2.imwrite(filepath, work_frame)
            print(f"  [{total_captured}] Saved: {filename}")

        if key in (ord('a'), ord('A')):
            auto_capture = not auto_capture
            status = "ON" if auto_capture else "OFF"
            print(f"\n[AUTO-CAPTURE] {status}")
            last_auto_time = time.time()

        elif key in (ord('q'), ord('Q'), 27):
            print("\n[INFO] Menutup program...")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("DATASET COLLECTION SELESAI")
    print("=" * 60)
    print(f"Total gambar tersimpan : {total_captured}")
    print(f"Folder output          : {os.path.abspath(DATASET_DIR)}")
    print(f"Ukuran gambar          : {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}")
    print("\nLangkah selanjutnya:")
    print("  1. Buka https://roboflow.com (gratis)")
    print("  2. Upload semua gambar dari folder di atas")
    print("  3. Labeling (gambar kotak di setiap kardus)")
    print("  4. Export dataset format YOLOv8")
    print("  5. Training di Google Colab menggunakan notebook yang sudah disiapkan")
    print("=" * 60)


if __name__ == '__main__':
    main()
