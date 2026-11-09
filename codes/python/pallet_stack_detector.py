# -*- coding: utf-8 -*-
"""
Pallet Stack Height & Arrangement Detector
Real-time monitoring via CCTV RTSP (Palletizing 6)
Mendeteksi:
  - Jumlah tumpukan/layer pada pallet
  - Susunan kardus per layer (valid/invalid)
  - Variant kardus (Sachet 20gr, YB77, BB725)
  - Progress penyusunan real-time
By: Yaser Ali Husen / Antigravity AI
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
from collections import Counter
import openpyxl
from ultralytics import YOLO

# ==================== THREADED VIDEO CAPTURE ====================
class ThreadedVideoCapture:
    """Membaca frame RTSP di thread terpisah agar tidak lag."""
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

# ==================== ROI SELECTION (MOUSE DRAW) ====================
class ROISelector:
    """Pilih area fokus (ROI) dengan menggambar kotak via mouse."""
    def __init__(self):
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.roi = None       # (x1, y1, x2, y2) final
        self.temp_frame = None

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
        """Tampilkan frame dan minta user menggambar kotak ROI.
        Return: (x1, y1, x2, y2) atau None jika di-skip."""
        self.roi = None
        self.temp_frame = frame.copy()
        win_name = 'PILIH AREA PALLET (Gambar Kotak dengan Mouse, ENTER = Konfirmasi, ESC = Full Frame)'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("PILIH AREA FOKUS (ROI)")
        print("=" * 60)
        print("  Gambar kotak di area pallet yang ingin dimonitor.")
        print("  Tekan ENTER untuk konfirmasi, ESC untuk skip (full frame).")
        print("=" * 60)

        while True:
            display = self.temp_frame.copy()

            # Gambar instruksi
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            cv2.putText(display,
                "GAMBAR KOTAK di area pallet | ENTER = OK | ESC = Skip (full frame)",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            # Gambar kotak saat sedang di-drag
            if self.drawing and self.start_pt and self.end_pt:
                cv2.rectangle(display, self.start_pt, self.end_pt,
                              (0, 255, 0), 2)

            # Gambar ROI final (setelah mouse up)
            if self.roi:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display, f"ROI: ({x1},{y1}) - ({x2},{y2})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                # Dim area di luar ROI
                mask = np.zeros_like(display)
                mask[y1:y2, x1:x2] = display[y1:y2, x1:x2]
                dimmed = cv2.addWeighted(display, 0.3, np.zeros_like(display), 0.7, 0)
                dimmed[y1:y2, x1:x2] = display[y1:y2, x1:x2]
                display = dimmed
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display, "Tekan ENTER untuk konfirmasi",
                            (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13 and self.roi:  # ENTER
                cv2.destroyWindow(win_name)
                print(f"[OK] ROI dipilih: {self.roi}")
                return self.roi
            elif key == 27:  # ESC = skip, gunakan full frame
                cv2.destroyWindow(win_name)
                print("[INFO] ROI di-skip. Menggunakan full frame.")
                return None


# ==================== CONFIGURATION ====================
# Sumber Video CCTV (Palletizing 6)
VIDEO_SOURCE = "rtsp://admin:1q2w3e4r@10.11.8.137:554/Streaming/Channels/101"

# Resolusi tampilan
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ROI Default (x1, y1, x2, y2) untuk Palletizing 6 (Pallet Tengah)
DEFAULT_ROI = (340, 220, 730, 710)

# Path Model YOLO Custom (variant classifier)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR, '..', 'pallet', 'runs', 'detect',
    'pallet_variant_detection', 'variant_classifier_v2',
    'weights', 'best.pt'
))

# Detection settings
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.35
YOLO_IMGSZ = 640

# Row/Layer grouping tolerance (pixel)
# Deteksi dengan Y-center yang berjarak <= tolerance dianggap 1 layer
ROW_TOLERANCE_PX = 40

# Konfigurasi variant (dari training summary)
VARIANT_CONFIG = {
    0: {
        'name': 'Sachet 20gr LunchBox',
        'short': 'JB',
        'box_per_layer': 14,
        'expected_layers': 6,
        'total_boxes': 84,
        'color': (0, 200, 80),       # Hijau
        'color_bright': (100, 255, 150),
    },
    1: {
        'name': 'Variant YB 77',
        'short': 'YB',
        'box_per_layer': 20,
        'expected_layers': 6,
        'total_boxes': 120,
        'color': (0, 165, 255),      # Orange
        'color_bright': (80, 200, 255),
    },
    2: {
        'name': 'Variant BB 725',
        'short': 'BB',
        'box_per_layer': 13,
        'expected_layers': 7,
        'total_boxes': 91,
        'color': (255, 100, 80),     # Biru-merah
        'color_bright': (255, 160, 140),
    }
}

# Fallback Varian Default / Unknown Box (Membaca kardus umum jika varian belum teridentifikasi)
DEFAULT_VARIANT_INFO = {
    'name': 'Kardus (Generic Box)',
    'short': 'BOX',
    'box_per_layer': 14,
    'expected_layers': 6,
    'total_boxes': 84,
    'color': (0, 255, 0),       # Hijau Neon
    'color_bright': (100, 255, 100),
}

def get_variant_info(cls):
    """Mendapatkan info varian. Jika cls None / tidak dikenal, gunakan fallback generic box."""
    if cls is not None and cls in VARIANT_CONFIG:
        return VARIANT_CONFIG[cls]
    return DEFAULT_VARIANT_INFO

# Output Excel
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCEL_FILE = os.path.join(OUTPUT_DIR,
    f"pallet_stack_{datetime.now().strftime('%Y%m%d')}.xlsx")

# ==================== EXCEL SETUP ====================
def init_excel():
    if not os.path.exists(EXCEL_FILE):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Pallet Stack Log"
        ws.append((
            "Timestamp", "Dominant Variant", "Detected Layers",
            "Expected Layers", "Total Box Detected", "Expected Total Box",
            "Layer Status", "Arrangement Status",
            "Status Overall", "Detail"
        ))
        wb.save(EXCEL_FILE)

def log_to_excel(variant_name, det_layers, exp_layers, total_box,
                 exp_total, layer_status, arrangement_status,
                 overall_status, detail):
    try:
        wb = openpyxl.load_workbook(EXCEL_FILE)
        ws = wb.active
        ws.append((
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            variant_name, det_layers, exp_layers, total_box, exp_total,
            layer_status, arrangement_status, overall_status, detail
        ))
        wb.save(EXCEL_FILE)
    except Exception as e:
        print(f"[ERROR] Gagal menulis Excel: {e}")

# ==================== LAYER GROUPING ====================
def group_detections_by_layer(detections, tolerance_px=ROW_TOLERANCE_PX):
    """
    Group bounding box detections into horizontal layers
    berdasarkan Y-center clustering.
    Return: list of layers (sorted dari BAWAH ke ATAS),
            setiap layer adalah list of detections.
    """
    if not detections:
        return []

    # Sort by Y-center (atas ke bawah)
    sorted_dets = sorted(detections,
                         key=lambda d: (d['y1'] + d['y2']) / 2)

    layers = []
    current_layer = [sorted_dets[0]]

    for det in sorted_dets[1:]:
        cy = (det['y1'] + det['y2']) / 2
        layer_cy = sum((d['y1'] + d['y2']) / 2
                       for d in current_layer) / len(current_layer)

        if abs(cy - layer_cy) <= tolerance_px:
            current_layer.append(det)
        else:
            layers.append(sorted(current_layer, key=lambda d: d['x1']))
            current_layer = [det]

    layers.append(sorted(current_layer, key=lambda d: d['x1']))

    # Sort layers dari BAWAH ke ATAS (y besar = bawah gambar = layer bawah)
    layers.sort(
        key=lambda r: sum((d['y1']+d['y2'])/2 for d in r) / len(r),
        reverse=True
    )
    return layers

# ==================== VALIDATION ====================
def get_dominant_variant(detections):
    """Tentukan variant dominan dari detections (weighted confidence)."""
    if not detections:
        return None, 0.0, {}

    class_weights = {}
    class_counts = {}
    for d in detections:
        cls = d['class']
        conf = d['confidence']
        class_weights[cls] = class_weights.get(cls, 0.0) + conf
        class_counts[cls] = class_counts.get(cls, 0) + 1

    best_cls = max(class_weights, key=class_weights.get)
    avg_conf = class_weights[best_cls] / class_counts[best_cls]

    distribution = {}
    total = sum(class_counts.values())
    for cls, cnt in class_counts.items():
        info = get_variant_info(cls)
        distribution[info.get('short', f'C{cls}')] = {
            'count': cnt,
            'pct': cnt / total * 100
        }

    return best_cls, avg_conf, distribution


def validate_pallet(detections, layers):
    """
    Validasi lengkap pallet:
    - Jumlah layer vs expected
    - Jumlah box per layer vs expected
    - Konsistensi variant (tidak tercampur)
    - Alignment / kerapian

    Returns: dict with validation results
    """
    if not detections or not layers:
        return {
            'valid': False,
            'layer_count': 0,
            'expected_layers': 0,
            'total_box': 0,
            'expected_total': 0,
            'dominant_variant': None,
            'dominant_conf': 0,
            'distribution': {},
            'layer_status': 'NO DATA',
            'arrangement_status': 'NO DATA',
            'overall_status': 'NO DATA',
            'errors': ['Tidak ada deteksi'],
            'layer_details': [],
        }

    # Dominant variant
    dom_cls, dom_conf, distribution = get_dominant_variant(detections)
    vconfig = get_variant_info(dom_cls)
    expected_layers = vconfig.get('expected_layers', 6)
    expected_bpl = vconfig.get('box_per_layer', 14)
    expected_total = vconfig.get('total_boxes', 84)

    detected_layers = len(layers)
    total_box = len(detections)

    errors = []
    layer_details = []

    # === Validasi per Layer ===
    mixed_variant_layers = 0
    wrong_count_layers = 0

    for i, layer in enumerate(layers):
        layer_cls = Counter([d['class'] for d in layer])
        layer_dominant = layer_cls.most_common(1)[0][0]
        layer_count = len(layer)
        is_mixed = len(layer_cls) > 1
        is_correct_count = layer_count == expected_bpl

        # Cek alignment (standard deviation X-center)
        if len(layer) >= 2:
            x_centers = [(d['x1'] + d['x2']) / 2 for d in layer]
            y_centers = [(d['y1'] + d['y2']) / 2 for d in layer]
            y_std = np.std(y_centers)
            is_aligned = y_std < ROW_TOLERANCE_PX
        else:
            is_aligned = True

        layer_info = get_variant_info(layer_dominant)
        detail = {
            'layer_num': i + 1,      # 1 = paling bawah
            'box_count': layer_count,
            'expected_count': expected_bpl,
            'dominant_class': layer_dominant,
            'dominant_name': layer_info.get('short', '?'),
            'is_mixed': is_mixed,
            'is_correct_count': is_correct_count,
            'is_aligned': is_aligned,
            'status': 'OK'
        }

        issues = []
        if is_mixed:
            mixed_variant_layers += 1
            issues.append('CAMPUR')
        if not is_correct_count:
            wrong_count_layers += 1
            diff = layer_count - expected_bpl
            if diff > 0:
                issues.append(f'+{diff}')
            else:
                issues.append(f'{diff}')
        if not is_aligned:
            issues.append('MIRING')

        if issues:
            detail['status'] = '/'.join(issues)

        layer_details.append(detail)

    # === Overall Status ===
    # Layer count status
    if detected_layers == expected_layers:
        layer_status = 'LENGKAP'
    elif detected_layers < expected_layers:
        layer_status = f'KURANG ({detected_layers}/{expected_layers})'
        errors.append(f'Layer kurang: {detected_layers}/{expected_layers}')
    else:
        layer_status = f'LEBIH ({detected_layers}/{expected_layers})'
        errors.append(f'Layer lebih: {detected_layers}/{expected_layers}')

    # Arrangement status
    if mixed_variant_layers > 0:
        arrangement_status = f'VARIANT CAMPUR ({mixed_variant_layers} layer)'
        errors.append(f'{mixed_variant_layers} layer variant tercampur')
    elif wrong_count_layers > 0:
        arrangement_status = f'JUMLAH SALAH ({wrong_count_layers} layer)'
        errors.append(f'{wrong_count_layers} layer jumlah box salah')
    else:
        arrangement_status = 'RAPI'

    is_valid = (len(errors) == 0 and detected_layers >= 1)
    overall = 'VALID' if is_valid else 'INVALID'

    return {
        'valid': is_valid,
        'layer_count': detected_layers,
        'expected_layers': expected_layers,
        'total_box': total_box,
        'expected_total': expected_total,
        'dominant_variant': dom_cls,
        'dominant_conf': dom_conf,
        'distribution': distribution,
        'layer_status': layer_status,
        'arrangement_status': arrangement_status,
        'overall_status': overall,
        'errors': errors,
        'layer_details': layer_details,
    }

# ==================== VISUAL OVERLAY ====================
def draw_glassy_panel(img, x, y, w, h, alpha=0.75):
    """Panel semi-transparan (glassmorphism effect)."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_progress_bar(img, x, y, w, h, progress, color, bg_color=(60,60,60)):
    """Progress bar visual."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)


def draw_overlay(frame, detections, layers, validation, capture_count, roi=None):
    """Gambar overlay dashboard + bounding box + layer markers + ROI border."""
    out = frame.copy()
    h, w = out.shape[:2]

    dom_cls = validation.get('dominant_variant')
    vconfig = VARIANT_CONFIG.get(dom_cls, {})
    variant_color = vconfig.get('color', (200, 200, 200))
    variant_color_bright = vconfig.get('color_bright', (255, 255, 255))

    # ── ROI Border (kotak hijau area fokus) ──
    if roi:
        rx1, ry1, rx2, ry2 = roi
        # Dim area di luar ROI
        mask_overlay = out.copy()
        cv2.rectangle(mask_overlay, (0, 0), (w, ry1), (0, 0, 0), -1)  # atas
        cv2.rectangle(mask_overlay, (0, ry2), (w, h), (0, 0, 0), -1)  # bawah
        cv2.rectangle(mask_overlay, (0, ry1), (rx1, ry2), (0, 0, 0), -1)  # kiri
        cv2.rectangle(mask_overlay, (rx2, ry1), (w, ry2), (0, 0, 0), -1)  # kanan
        cv2.addWeighted(mask_overlay, 0.4, out, 0.6, 0, out)
        # Border ROI
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(out, "ROI", (rx1 + 5, ry1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # ── Bounding Box per Deteksi ──
    for det in detections:
        x1, y1 = int(det['x1']), int(det['y1'])
        x2, y2 = int(det['x2']), int(det['y2'])
        cls = det['class']
        conf = det['confidence']
        info = get_variant_info(cls)
        color = info.get('color', (200, 200, 200))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{info.get('short', 'BOX')} {conf:.0%}"
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(out, (x1, y1 - lsz[1] - 6), (x1 + lsz[0] + 4, y1),
                      color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # ── Outer Bounding Box (Membungkus 1 Tumpukan Pallet / Block) ──
    if detections:
        min_x = min(d['x1'] for d in detections)
        min_y = min(d['y1'] for d in detections)
        max_x = max(d['x2'] for d in detections)
        max_y = max(d['y2'] for d in detections)

        pad = 8
        ox1 = max(0, int(min_x - pad))
        oy1 = max(0, int(min_y - pad))
        ox2 = min(w, int(max_x + pad))
        oy2 = min(h, int(max_y + pad))

        # Tentukan status inspection badge
        ov_status = validation.get('overall_status', 'NO DATA')
        det_layers = validation.get('layer_count', 0)
        exp_layers = validation.get('expected_layers', 6)
        tot_box = validation.get('total_box', 0)

        if ov_status == 'VALID':
            badge_color = (0, 255, 0)
            badge_title = "INSPECTION: SUCCESS"
            badge_sub = f"STATUS: ACCEPTED ({det_layers}/{exp_layers} Layer | {tot_box} Pcs)"
        elif ov_status == 'NO DATA':
            badge_color = (0, 220, 255)
            badge_title = "INSPECTION: DETECTING..."
            badge_sub = "STATUS: INITIALIZING"
        else:
            if det_layers < exp_layers:
                badge_color = (0, 220, 255)  # Cyan/Yellow saat menyusun
                badge_title = "INSPECTION: IN PROGRESS"
                badge_sub = f"STATUS: STACKING ({det_layers}/{exp_layers} Layer | {tot_box} Pcs)"
            else:
                badge_color = (0, 0, 255)    # Merah jika ada kesalahan/anomaly
                badge_title = "INSPECTION: WARNING / ANOMALY"
                badge_sub = f"STATUS: CHECK ARRANGEMENT ({det_layers}/{exp_layers} Layer)"

        # Gambar bingkai utama pembungkus 1 tumpukan pallet (Outer 3D Box)
        cv2.rectangle(out, (ox1, oy1), (ox2, oy2), badge_color, 3)

        # Gambar aksen 3D corner notches pada sudut bingkai
        corner_len = min(25, max(5, (ox2 - ox1) // 4), max(5, (oy2 - oy1) // 4))
        cv2.line(out, (ox1, oy1), (ox1 + corner_len, oy1), (255, 255, 255), 4)
        cv2.line(out, (ox1, oy1), (ox1, oy1 + corner_len), (255, 255, 255), 4)
        cv2.line(out, (ox2, oy1), (ox2 - corner_len, oy1), (255, 255, 255), 4)
        cv2.line(out, (ox2, oy1), (ox2, oy1 + corner_len), (255, 255, 255), 4)

        cv2.line(out, (ox1, oy2), (ox1 + corner_len, oy2), (255, 255, 255), 4)
        cv2.line(out, (ox1, oy2), (ox1, oy2 - corner_len), (255, 255, 255), 4)
        cv2.line(out, (ox2, oy2), (ox2 - corner_len, oy2), (255, 255, 255), 4)
        cv2.line(out, (ox2, oy2), (ox2, oy2 - corner_len), (255, 255, 255), 4)

        # Gambar Inspection Badge di atas tumpukan pallet (seperti gambar referensi)
        badge_w = 330
        badge_h = 42
        bx1 = max(5, ox1)
        by1 = max(5, oy1 - badge_h - 8)

        # Panel latar belakang badge
        badge_bg = out.copy()
        cv2.rectangle(badge_bg, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (10, 10, 10), -1)
        cv2.addWeighted(badge_bg, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (bx1, by1), (bx1 + badge_w, by1 + badge_h), badge_color, 2)

        cv2.putText(out, badge_title, (bx1 + 10, by1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 2, cv2.LINE_AA)
        cv2.putText(out, badge_sub, (bx1 + 10, by1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Garis Pemisah Layer + Label ──
    for i, layer in enumerate(layers):
        if not layer:
            continue
        ys = [(d['y1'] + d['y2']) / 2 for d in layer]
        row_cy = int(sum(ys) / len(ys))
        layer_dominant = Counter([d['class'] for d in layer]).most_common(1)[0][0]
        linfo = VARIANT_CONFIG.get(layer_dominant, {})
        lcolor = linfo.get('color', (200, 200, 200))

        # Garis horizontal (hanya di lebar tumpukan pallet)
        if layer:
            min_lx = int(min(d['x1'] for d in layer))
            max_lx = int(max(d['x2'] for d in layer))
            lx1 = max(5, min_lx - 10)
            lx2 = min(w - 5, max_lx + 10)
        elif roi:
            lx1, lx2 = roi[0], roi[2]
        else:
            lx1, lx2 = 5, w - 5
        cv2.line(out, (lx1, row_cy), (lx2, row_cy), lcolor, 1, cv2.LINE_AA)

        # Label layer
        detail = validation['layer_details'][i] if i < len(validation['layer_details']) else {}
        status = detail.get('status', '?')
        count = detail.get('box_count', 0)
        status_color = (0, 220, 80) if status == 'OK' else (0, 0, 255)

        tag = f"L{i+1}: {count}pcs [{status}]"
        cv2.putText(out, tag, (lx1 + 3, row_cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

    # ── Height Gauge (Bar Vertikal di Kanan) ──
    if layers and dom_cls is not None:
        exp_layers = validation['expected_layers']
        det_layers = validation['layer_count']
        gauge_x = w - 45
        gauge_y_top = 160
        gauge_y_bot = h - 50
        gauge_h = gauge_y_bot - gauge_y_top
        gauge_w = 25

        # Background gauge
        draw_glassy_panel(out, gauge_x - 5, gauge_y_top - 20,
                          gauge_w + 10, gauge_h + 40, 0.7)

        cv2.putText(out, "STACK", (gauge_x - 3, gauge_y_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Fill berdasarkan progress layer
        progress = min(det_layers / max(exp_layers, 1), 1.0)
        fill_h = int(gauge_h * progress)

        # Background
        cv2.rectangle(out, (gauge_x, gauge_y_top),
                      (gauge_x + gauge_w, gauge_y_bot), (40, 40, 40), -1)
        # Filled dari bawah ke atas
        if fill_h > 0:
            cv2.rectangle(out,
                          (gauge_x, gauge_y_bot - fill_h),
                          (gauge_x + gauge_w, gauge_y_bot),
                          variant_color, -1)
        # Border
        cv2.rectangle(out, (gauge_x, gauge_y_top),
                      (gauge_x + gauge_w, gauge_y_bot), (120, 120, 120), 1)

        # Teks jumlah layer
        cv2.putText(out, f"{det_layers}/{exp_layers}",
                    (gauge_x - 5, gauge_y_bot + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, variant_color_bright, 1)

    # ── Panel Dashboard Atas ──
    panel_h = 150
    draw_glassy_panel(out, 0, 0, w - 55, panel_h, 0.80)

    # Status VALID / INVALID (pojok kanan atas panel)
    ov_status = validation['overall_status']
    if ov_status == 'VALID':
        st_color = (0, 220, 80)
    elif ov_status == 'NO DATA':
        st_color = (120, 120, 120)
    else:
        st_color = (0, 0, 255)

    cv2.putText(out, ov_status, (w - 230, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, st_color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (w - 238, 10), (w - 65, 48), st_color, 2)

    # Judul
    cv2.putText(out, "PALLET STACK DETECTOR",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(out, "Palletizing 6 | RTSP Live",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (140, 140, 140), 1, cv2.LINE_AA)

    # Variant & confidence
    if dom_cls is not None:
        vname = vconfig.get('name', '?')
        conf_pct = validation['dominant_conf']
        cv2.putText(out,
                    f"Variant: {vname}  (conf: {conf_pct:.0%})",
                    (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    variant_color_bright, 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "Menunggu deteksi...",
                    (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 100, 100), 1, cv2.LINE_AA)

    # Layer info
    det_l = validation['layer_count']
    exp_l = validation['expected_layers']
    tot_b = validation['total_box']
    exp_t = validation['expected_total']

    cv2.putText(out,
                f"Layer: {det_l}/{exp_l}   |   Box: {tot_b}/{exp_t}",
                (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Progress bar
    if exp_l > 0:
        progress = min(det_l / exp_l, 1.0)
        bar_color = (0, 220, 80) if progress >= 1.0 else variant_color
        draw_progress_bar(out, 12, 100, 350, 14, progress, bar_color)
        pct_text = f"{progress:.0%}"
        cv2.putText(out, pct_text, (370, 113),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1, cv2.LINE_AA)

    # Status detail
    ls = validation['layer_status']
    ar = validation['arrangement_status']
    cv2.putText(out,
                f"Layer: {ls}  |  Susunan: {ar}",
                (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                st_color, 1, cv2.LINE_AA)

    # Errors
    if validation['errors']:
        err_text = " | ".join(validation['errors'][:3])
        cv2.putText(out, f"! {err_text}",
                    (12, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (0, 0, 255), 1, cv2.LINE_AA)

    # ── Panel Bawah ──
    cv2.rectangle(out, (0, h - 30), (w, h), (25, 25, 25), -1)
    cv2.putText(out,
        f"[SPACE] Capture  [S] Screenshot  [D] Re-draw ROI  [R] Reset  [Q] Quit  |  #{capture_count}",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("SISTEM DETEKSI TINGGI & VALIDASI SUSUNAN PALLET")
    print("=" * 60)

    # 1. Init Excel
    init_excel()
    print(f"[INFO] Log Excel: {os.path.abspath(EXCEL_FILE)}")

    # 2. Load YOLO Model
    print(f"[INFO] Loading model: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        print("Pastikan model best.pt ada di folder yang benar.")
        return

    try:
        model = YOLO(MODEL_PATH)
        print("[OK] Model YOLO berhasil dimuat!")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return

    # 3. Connect to CCTV
    print(f"[INFO] Menghubungkan ke CCTV: {VIDEO_SOURCE}...")
    cap = ThreadedVideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Gagal terhubung ke CCTV!")
        return

    print("[OK] CCTV Palletizing 6 terhubung!")

    # 4. Ambil 1 frame pertama untuk ROI selection
    time.sleep(1)  # Tunggu frame stabil
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame pertama!")
        cap.release()
        return
    first_frame = cv2.resize(first_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Mode ROI (DEFAULT_ROI=None -> Full Frame tanpa zoom saat awal)
    if DEFAULT_ROI is not None:
        active_roi = DEFAULT_ROI
        print(f"[OK] Menggunakan Presets ROI: {active_roi}")
    else:
        active_roi = None
        print("[INFO] Mode Full Frame Aktif (Tanpa Zoom). Tekan 'D' jika ingin menggambar ROI.")

    print("\n[PETUNJUK] SPACE=Capture | S=Screenshot | D=Re-draw ROI | R=Reset | Q=Quit\n")
    print("=" * 60)

    cv2.namedWindow('Pallet Stack Detector - Palletizing 6', cv2.WINDOW_NORMAL)

    capture_count = 0
    start_time = datetime.now()
    empty_validation = {
        'valid': False, 'layer_count': 0, 'expected_layers': 0,
        'total_box': 0, 'expected_total': 0,
        'dominant_variant': None, 'dominant_conf': 0,
        'distribution': {}, 'layer_status': 'NO DATA',
        'arrangement_status': 'NO DATA', 'overall_status': 'NO DATA',
        'errors': [], 'layer_details': [],
    }
    last_validation = dict(empty_validation)
    last_detections = []
    last_layers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame tidak terbaca, mencoba ulang...")
            time.sleep(0.1)
            continue

        # Resize
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # 5. YOLO Inference
        detections = []
        if active_roi:
            rx1, ry1, rx2, ry2 = active_roi
            roi_crop = frame[ry1:ry2, rx1:rx2]
            results = model(roi_crop, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                            verbose=False, imgsz=YOLO_IMGSZ)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'x1': float(bx1 + rx1), 'y1': float(by1 + ry1),
                            'x2': float(bx2 + rx1), 'y2': float(by2 + ry1),
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls[0])
                        })
        else:
            results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                            verbose=False, imgsz=YOLO_IMGSZ)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'x1': float(x1), 'y1': float(y1),
                            'x2': float(x2), 'y2': float(y2),
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls[0])
                        })

        # 6. Group into layers & validate
        if detections:
            layers = group_detections_by_layer(detections)
            validation = validate_pallet(detections, layers)
            last_detections = detections
            last_layers = layers
            last_validation = validation
        else:
            last_detections = []
            last_layers = []

        # 7. Draw overlay pada frame CCTV asli (tanpa distorsi)
        frame_display = draw_overlay(
            frame, last_detections, last_layers,
            last_validation, capture_count, roi=active_roi
        )

        cv2.imshow('Pallet Stack Detector - Palletizing 6', frame_display)

        # 8. Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # CAPTURE & LOG
            if last_detections:
                capture_count += 1
                v = last_validation
                dom_cls = v['dominant_variant']
                vinfo = VARIANT_CONFIG.get(dom_cls, {})

                log_to_excel(
                    vinfo.get('name', '?'),
                    v['layer_count'], v['expected_layers'],
                    v['total_box'], v['expected_total'],
                    v['layer_status'], v['arrangement_status'],
                    v['overall_status'],
                    '; '.join(v['errors']) if v['errors'] else 'OK'
                )

                # Save screenshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ss_path = os.path.join(OUTPUT_DIR,
                                       f"pallet_capture_{ts}.jpg")
                cv2.imwrite(ss_path, frame_display)

                print(f"\n[CAPTURE #{capture_count}]")
                print(f"  Variant : {vinfo.get('name', '?')} "
                      f"(conf: {v['dominant_conf']:.0%})")
                print(f"  Layer   : {v['layer_count']}/{v['expected_layers']}")
                print(f"  Box     : {v['total_box']}/{v['expected_total']}")
                print(f"  Status  : {v['overall_status']} - "
                      f"{v['layer_status']} | {v['arrangement_status']}")
                if v['errors']:
                    for err in v['errors']:
                        print(f"  ! {err}")
                print(f"  Saved   : {ss_path}")
            else:
                print("\n[INFO] Tidak ada deteksi untuk di-capture.")

        elif key in (ord('d'), ord('D')):
            # RE-DRAW ROI
            ret2, redraw_frame = cap.read()
            if ret2:
                redraw_frame = cv2.resize(redraw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                roi_selector2 = ROISelector()
                new_roi = roi_selector2.select(redraw_frame)
                active_roi = new_roi
                last_detections = []
                last_layers = []
                last_validation = dict(empty_validation)
                if active_roi:
                    print(f"[OK] ROI baru: {active_roi}")
                else:
                    print("[INFO] ROI dihapus. Menggunakan full frame.")

        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ss_path = os.path.join(OUTPUT_DIR, f"screenshot_{ts}.jpg")
            cv2.imwrite(ss_path, frame_display)
            print(f"\n[SCREENSHOT] Saved: {ss_path}")

        elif key in (ord('r'), ord('R')):
            capture_count = 0
            last_detections = []
            last_layers = []
            last_validation = dict(empty_validation)
            print("\n[RESET] Counter & deteksi di-reset.")

        elif key in (ord('q'), ord('Q'), 27):
            print("\n[INFO] Menutup program...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    elapsed = datetime.now() - start_time
    elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed.total_seconds()))

    print("\n" + "=" * 60)
    print("REKAP AKHIR - PALLET STACK DETECTOR")
    print("=" * 60)
    print(f"Total Capture     : {capture_count}")
    print(f"Durasi Monitoring : {elapsed_str}")
    print(f"Excel Log         : {os.path.abspath(EXCEL_FILE)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
