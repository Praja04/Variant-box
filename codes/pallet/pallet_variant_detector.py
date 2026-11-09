# -*- coding: utf-8 -*-
"""
Pallet Variant Detector - FIXED
Fix: Nama variant sync dengan collector (JB/BB/YB), CAP_DSHOW, class name mapping
By: Yaser Ali Husen (Fixed)
"""

import cv2
import numpy as np
from datetime import datetime
import openpyxl
import os
from ultralytics import YOLO

# ============= CONFIGURATION =============
# FIX: Nama variant HARUS sama persis dengan class names di data.yaml
# dan class_id harus sesuai dengan urutan di data.yaml
# data.yaml: names: ['Variant_JB', 'Variant_BB', 'Variant_YB']
VARIANT_CONFIG = {
    0: {   # class 0 = Variant_JB
        'name':        'Variant JB',
        'description': '3x4 Grid (12 boxes)',
        'total_boxes': 12,
        'grid':        {'rows': 3, 'cols': 4},
        'color':       (0, 255, 0)   # Hijau
    },
    1: {   # class 1 = Variant_BB
        'name':        'Variant BB',
        'description': '4x5 Grid (20 boxes)',
        'total_boxes': 20,
        'grid':        {'rows': 4, 'cols': 5},
        'color':       (255, 0, 0)   # Biru
    },
    2: {   # class 2 = Variant_YB
        'name':        'Variant YB',
        'description': '2x7 Grid (14 boxes)',
        'total_boxes': 14,
        'grid':        {'rows': 2, 'cols': 7},
        'color':       (0, 165, 255) # Orange
    }
}

# FIX: Path model - sesuaikan jika training selesai
MODEL_PATH = 'runs/detect/pallet_variant_detection/variant_classifier_v1/weights/best.pt'

if not os.path.exists(MODEL_PATH):
    print(f"⚠ Custom model tidak ditemukan: {MODEL_PATH}")
    print("  Menggunakan pretrained model (hanya untuk demo).")
    print("  Jalankan train_variant_model.py terlebih dahulu!\n")
    MODEL_PATH = 'yolov8n.pt'

CONFIDENCE_THRESHOLD = 0.35
OUTPUT_PATH = "../output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============= GLOBAL =============
total_detections = 0
variant_counts   = {0: 0, 1: 0, 2: 0}
last_detection   = None

# ============= EXCEL SETUP =============
excel_file = os.path.join(OUTPUT_PATH,
    f"variant_detection_{datetime.now().strftime('%Y%m%d')}.xlsx")

if not os.path.exists(excel_file):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(("Timestamp", "Detected Variant", "Class ID", "Confidence",
                "Expected Boxes", "Notes"))
    wb.save(excel_file)

def log_to_excel(variant_id, confidence, notes=""):
    wb  = openpyxl.load_workbook(excel_file)
    ws  = wb.active
    info = VARIANT_CONFIG.get(variant_id, {})
    ws.append((
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        info.get('name', f'Unknown (class {variant_id})'),
        variant_id,
        f"{confidence:.2%}",
        info.get('total_boxes', 'N/A'),
        notes
    ))
    wb.save(excel_file)

def draw_detection_overlay(frame, detections):
    frame_out = frame.copy()
    height, width = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        cls        = det['class']
        conf       = det['confidence']
        info       = VARIANT_CONFIG.get(cls, {})
        color      = info.get('color', (255, 255, 255))
        label      = f"{info.get('name', f'Class {cls}')} ({conf:.2%})"

        cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 3)

        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_out,
                      (x1, y1 - lsz[1] - 10), (x1 + lsz[0] + 10, y1),
                      color, -1)
        cv2.putText(frame_out, label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(frame_out, info.get('description', ''),
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Panel info atas
    panel_h = 185
    cv2.rectangle(frame_out, (10, 10), (width - 10, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame_out, (10, 10), (width - 10, panel_h), (255, 255, 255), 2)

    y = 38
    cv2.putText(frame_out, "PALLET VARIANT DETECTOR",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y += 32
    if detections:
        main = max(detections, key=lambda d: d['confidence'])
        cls  = main['class']
        info = VARIANT_CONFIG.get(cls, {})
        color = info.get('color', (255, 255, 255))

        cv2.putText(frame_out, f"Detected: {info.get('name', 'Unknown')}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28
        cv2.putText(frame_out, f"Confidence : {main['confidence']:.2%}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame_out, f"Expected Boxes: {info.get('total_boxes', 'N/A')}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame_out, "Tidak ada variant terdeteksi",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        y += 25

    y += 38
    cv2.putText(frame_out, f"Total Detections: {total_detections}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 22
    stats = (f"JB: {variant_counts[0]}  |  BB: {variant_counts[1]}  "
             f"|  YB: {variant_counts[2]}")
    cv2.putText(frame_out, stats,
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Panel kontrol bawah
    by = height - 65
    cv2.rectangle(frame_out, (10, by - 5), (width - 10, height - 10), (50, 50, 50), -1)
    cv2.putText(frame_out,
                "[SPACE] Capture & Log  [R] Reset Stats  [S] Screenshot  [Q] Quit",
                (20, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame_out,
                "Posisikan pallet di tengah frame untuk hasil deteksi terbaik",
                (20, by + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return frame_out

# ============= MAIN =============
print("=" * 60)
print("PALLET VARIANT DETECTOR - FIXED")
print("=" * 60)
print(f"\nLoading model: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded!")
    if hasattr(model, 'names'):
        print(f"  Model classes: {model.names}")
        # FIX: Verifikasi bahwa nama class sesuai yang diharapkan
        expected = ['Variant_JB', 'Variant_BB', 'Variant_YB']
        actual   = list(model.names.values()) if hasattr(model.names, 'values') else model.names
        if actual != expected and MODEL_PATH != 'yolov8n.pt':
            print(f"\n⚠ Peringatan: nama class model = {actual}")
            print(f"  Diharapkan                    = {expected}")
            print("  Pastikan model ditraining dari dataset yang benar!\n")
except Exception as e:
    print(f"✗ Gagal load model: {e}")
    exit(1)

print("\nOpening camera...")
# FIX: Pakai CAP_DSHOW untuk Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("✗ Gagal buka camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"✓ Camera ready: "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("\n[SPACE] Capture | [R] Reset | [S] Screenshot | [Q] Quit\n")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Live detection setiap frame
    results    = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': float(box.conf[0]),
                'class':      int(box.cls[0])
            })

    if detections:
        last_detection = detections

    frame_display = draw_detection_overlay(frame, detections)
    cv2.imshow('Pallet Variant Detector', frame_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if last_detection:
            main = max(last_detection, key=lambda d: d['confidence'])
            cls  = main['class']
            conf = main['confidence']

            total_detections += 1
            variant_counts[cls] = variant_counts.get(cls, 0) + 1

            log_to_excel(cls, conf, notes=f"{len(last_detection)} detection(s)")

            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            spath = os.path.join(OUTPUT_PATH, f"detection_{ts}.jpg")
            cv2.imwrite(spath, frame_display)

            info = VARIANT_CONFIG.get(cls, {})
            print(f"\n✓ CAPTURED → {info.get('name', 'Unknown')}"
                  f"  conf={conf:.2%}"
                  f"  boxes={info.get('total_boxes', '?')}"
                  f"  total={total_detections}")
        else:
            print("\n⚠ Tidak ada deteksi!")

    elif key in (ord('r'), ord('R')):
        total_detections = 0
        variant_counts   = {0: 0, 1: 0, 2: 0}
        print("\n✓ Statistik di-reset")

    elif key in (ord('s'), ord('S')):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        spath = os.path.join(OUTPUT_PATH, f"screenshot_{ts}.jpg")
        cv2.imwrite(spath, frame_display)
        print(f"\n📸 Screenshot: {spath}")

    elif key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("DETECTION SUMMARY")
print("="*60)
print(f"Total Detections: {total_detections}")
for cid, cnt in variant_counts.items():
    if cnt > 0:
        info = VARIANT_CONFIG.get(cid, {})
        pct  = cnt / total_detections * 100 if total_detections > 0 else 0
        print(f"  {info.get('name', f'Class {cid}')}: {cnt} ({pct:.1f}%)")
print(f"\nLog: {excel_file}")
print("="*60)