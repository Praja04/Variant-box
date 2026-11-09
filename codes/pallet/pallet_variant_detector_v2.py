# -*- coding: utf-8 -*-
"""
Pallet Variant Detector v2 — YOLO26 Edition
Upgrade dari YOLOv8 ke YOLO26 (NMS-Free, MuSGD, STAL)
Dengan:
  - Weighted Voting (confidence-based smoothing)
  - Test-Time Augmentation (TTA) toggle [A]
  - FP16 inference jika GPU tersedia
Variant:
  [0] Sachet_20gr_LunchBox : 14 box/layer x 6 layer = 84 box
  [1] Variant_YB77         : 20 box/layer x 6 layer = 120 box
  [2] Variant_BB725        : 13 box/layer x 7 layer = 91 box
By: Yaser Ali Husen (v2 → YOLO26 upgrade)
"""

import cv2
import numpy as np
from datetime import datetime
from collections import deque, Counter
import openpyxl
import os
from ultralytics import YOLO

# ============= CONFIGURATION =============
VARIANT_CONFIG = {
    0: {
        'name':          'Sachet_20gr_LunchBox',
        'description':   'Sachet 20gr Lunch Box',
        'box_per_layer': 14,
        'total_layers':  6,
        'total_boxes':   84,
        'color':         (0, 200, 80)
    },
    1: {
        'name':          'Variant_YB77',
        'description':   'Variant YB 77',
        'box_per_layer': 20,
        'total_layers':  6,
        'total_boxes':   120,
        'color':         (0, 165, 255)
    },
    2: {
        'name':          'Variant_BB725',
        'description':   'Variant BB 725',
        'box_per_layer': 13,
        'total_layers':  7,
        'total_boxes':   91,
        'color':         (80, 80, 255)
    }
}

# ── Aturan susunan palet yang VALID ──────────────────────────
# Sesuaikan expected_sequence dengan aturan gudang/produksi kamu.
# Urutan class_id dari BAWAH ke ATAS palet.
# Contoh di bawah: layer bawah BB725 (2), layer atas YB77 (1)
STACKING_RULES = {
    'mode':              'sequence',  # 'sequence' = urutan tertentu, 'any' = bebas selama sama
    'expected_sequence': [2, 1],      # BB725 bawah → YB77 atas  ← SESUAIKAN INI
    'allow_single':      True,        # True = 1 variant saja dianggap valid
    'min_confidence':    0.5,
    'row_tolerance_px':  60,
}

MODEL_PATH = 'runs/detect/pallet_variant_detection/variant_classifier_yolo26/weights/best.pt'
if not os.path.exists(MODEL_PATH):
    # Fallback ke model v2 lama jika YOLO26 belum di-train
    MODEL_PATH_FALLBACK = 'runs/detect/pallet_variant_detection/variant_classifier_v2/weights/best.pt'
    if os.path.exists(MODEL_PATH_FALLBACK):
        MODEL_PATH = MODEL_PATH_FALLBACK
        print(f"  [WARN] Model YOLO26 belum ada, fallback ke: {MODEL_PATH}")
        print(f"  [WARN] Jalankan train_variant_model_v2.py untuk training YOLO26!")
    else:
        print(f"  Model tidak ditemukan di:")
        print(f"    - runs/detect/pallet_variant_detection/variant_classifier_yolo26/weights/best.pt")
        print(f"    - {MODEL_PATH_FALLBACK}")
        print("  Pastikan path MODEL_PATH sesuai dengan lokasi best.pt kamu.")
        exit(1)

CONFIDENCE_THRESHOLD = 0.45     # ↑ dari 0.40 — YOLO26 confidence lebih terpercaya
SMOOTHING_FRAMES     = 7        # ↑ dari 5  — lebih stabil
ENABLE_TTA           = False    # Test-Time Augmentation (toggle via [A])
ENABLE_FP16          = True     # FP16 inference — auto-disable jika CPU
OUTPUT_PATH          = "../output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============= GLOBAL STATE =============
total_detections = 0
variant_counts   = {0: 0, 1: 0, 2: 0}
detection_buffer = deque(maxlen=SMOOTHING_FRAMES)

# ============= EXCEL SETUP =============
excel_file = os.path.join(OUTPUT_PATH,
    f"variant_detection_{datetime.now().strftime('%Y%m%d')}.xlsx")
if not os.path.exists(excel_file):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append((
        "Timestamp", "Detected Variant", "Class ID", "Confidence",
        "Box/Layer", "Total Layers", "Total Boxes",
        "Stacking Valid", "Stacking Detail", "Notes"
    ))
    wb.save(excel_file)

def log_to_excel(variant_id, confidence, valid, detail, notes=""):
    wb   = openpyxl.load_workbook(excel_file)
    ws   = wb.active
    info = VARIANT_CONFIG.get(variant_id, {})
    ws.append((
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        info.get('name', f'Unknown ({variant_id})'),
        variant_id,
        f"{confidence:.2%}",
        info.get('box_per_layer', 'N/A'),
        info.get('total_layers',  'N/A'),
        info.get('total_boxes',   'N/A'),
        "VALID" if valid else "INVALID",
        detail,
        notes
    ))
    wb.save(excel_file)

# ============= WEIGHTED SMOOTHING =============
def get_smoothed_detections(new_detections):
    """Smoothing dengan weighted voting berdasarkan confidence."""
    detection_buffer.append(new_detections)
    if not detection_buffer:
        return []
    all_classes = []
    for frame_dets in detection_buffer:
        for d in frame_dets:
            if d['confidence'] >= STACKING_RULES['min_confidence']:
                all_classes.append(d['class'])
    return all_classes

def get_weighted_dominant_class(detections):
    """Menentukan kelas dominan menggunakan weighted voting (confidence-based).
    Lebih akurat daripada simple counting karena deteksi dengan
    confidence tinggi punya bobot lebih besar."""
    if not detections:
        return None, 0.0
    # Akumulasi confidence per kelas
    class_weights = {}
    class_counts  = {}
    for d in detections:
        cls  = d['class']
        conf = d['confidence']
        class_weights[cls] = class_weights.get(cls, 0.0) + conf
        class_counts[cls]  = class_counts.get(cls, 0) + 1
    # Kelas dengan total confidence tertinggi menang
    best_cls = max(class_weights, key=class_weights.get)
    avg_conf = class_weights[best_cls] / class_counts[best_cls]
    return best_cls, avg_conf

# ============= STACKING VALIDATION =============
def group_detections_by_row(detections, tolerance_px=None):
    if tolerance_px is None:
        tolerance_px = STACKING_RULES['row_tolerance_px']
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: (d['y1'] + d['y2']) / 2)
    rows        = []
    current_row = [sorted_dets[0]]

    for det in sorted_dets[1:]:
        cy     = (det['y1'] + det['y2']) / 2
        row_cy = sum((d['y1'] + d['y2']) / 2 for d in current_row) / len(current_row)
        if abs(cy - row_cy) <= tolerance_px:
            current_row.append(det)
        else:
            rows.append(sorted(current_row, key=lambda d: d['x1']))
            current_row = [det]

    rows.append(sorted(current_row, key=lambda d: d['x1']))
    # Sort dari bawah ke atas (y besar = bawah)
    rows.sort(key=lambda r: sum((d['y1']+d['y2'])/2 for d in r) / len(r), reverse=True)
    return rows

def validate_stacking(detections):
    confident_dets = [d for d in detections
                      if d['confidence'] >= STACKING_RULES['min_confidence']]
    if not confident_dets:
        return False, "Tidak ada deteksi dengan confidence cukup", []

    rows = group_detections_by_row(confident_dets)
    if not rows:
        return False, "Tidak ada baris terdeteksi", []

    mode = STACKING_RULES['mode']

    # Single variant
    if STACKING_RULES['allow_single'] and len(rows) == 1:
        cls = Counter([d['class'] for d in rows[0]])
        if len(cls) == 1:
            only_cls = list(cls.keys())[0]
            info     = VARIANT_CONFIG.get(only_cls, {})
            n        = sum(cls.values())
            bpl      = info.get('box_per_layer', '?')
            return (True,
                    f"Single: {info.get('name','?')} — {n} kardus "
                    f"(valid layer: kelipatan {bpl})",
                    rows)

    # Mode sequence
    if mode == 'sequence':
        expected = STACKING_RULES['expected_sequence']
        if len(rows) < len(expected):
            return (False,
                    f"Kurang layer: {len(rows)} baris, diharapkan {len(expected)}",
                    rows)
        issues = []
        for i, (row, exp_cls) in enumerate(zip(rows, expected)):
            dominant = Counter([d['class'] for d in row]).most_common(1)[0][0]
            exp_info = VARIANT_CONFIG.get(exp_cls, {})
            act_info = VARIANT_CONFIG.get(dominant, {})
            if dominant != exp_cls:
                issues.append(
                    f"Layer {i+1}: expect {exp_info.get('name','?')} "
                    f"-> dapat {act_info.get('name','?')}"
                )
        if issues:
            return False, " | ".join(issues), rows
        return True, f"{len(rows)} layer sesuai urutan yang ditentukan", rows

    # Mode any
    all_classes = [d['class'] for row in rows for d in row]
    dominant    = Counter(all_classes).most_common(1)[0]
    if dominant[1] == len(all_classes):
        info = VARIANT_CONFIG.get(dominant[0], {})
        return True, f"Semua {len(all_classes)} kardus = {info.get('name','?')}", rows
    mixed  = Counter(all_classes)
    detail = ", ".join(
        f"{VARIANT_CONFIG.get(c,{}).get('name','?')}: {n}"
        for c, n in mixed.items()
    )
    return False, f"Variant tercampur: {detail}", rows

# ============= OVERLAY =============
def draw_overlay(frame, detections, valid, valid_detail, rows):
    out  = frame.copy()
    h, w = out.shape[:2]

    # Bounding box per deteksi
    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        cls   = det['class']
        conf  = det['confidence']
        info  = VARIANT_CONFIG.get(cls, {})
        color = info.get('color', (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{info.get('name','?')} {conf:.0%}"
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1, y1 - lsz[1] - 8), (x1 + lsz[0] + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    # Garis pemisah baris
    for i, row in enumerate(rows):
        if not row: continue
        ys      = [(d['y1'] + d['y2']) / 2 for d in row]
        row_cy  = int(sum(ys) / len(ys))
        row_cls = Counter([d['class'] for d in row]).most_common(1)[0][0]
        color   = VARIANT_CONFIG.get(row_cls, {}).get('color', (200, 200, 200))
        cv2.line(out, (5, row_cy), (w - 5, row_cy), color, 1)
        cv2.putText(out, f"L{i+1}", (5, row_cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Panel atas
    panel_h = 130
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    valid_color = (0, 200, 80) if valid else (60, 60, 255)
    valid_text  = "VALID" if valid else "INVALID"
    cv2.putText(out, valid_text, (w - 160, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, valid_color, 2)
    cv2.rectangle(out, (w - 168, 10), (w - 10, 52), valid_color, 2)

    cv2.putText(out, "PALLET DETECTOR v2 [YOLO26]",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)

    if detections:
        main  = max(detections, key=lambda d: d['confidence'])
        info  = VARIANT_CONFIG.get(main['class'], {})
        color = info.get('color', (255, 255, 255))
        bpl   = info.get('box_per_layer', '?')
        tl    = info.get('total_layers',  '?')
        tb    = info.get('total_boxes',   '?')
        cv2.putText(out,
            f"{info.get('name','?')}  conf={main['confidence']:.0%}  "
            f"| {bpl}box/layer x {tl}layer = {tb}box",
            (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    else:
        cv2.putText(out, "Tidak ada deteksi",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    detail_short = valid_detail[:80] + "..." if len(valid_detail) > 80 else valid_detail
    cv2.putText(out, detail_short,
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 200, 80) if valid else (80, 80, 255), 1)

    stats = (f"Total: {total_detections}  |  "
             f"Sachet:{variant_counts[0]}  "
             f"YB77:{variant_counts[1]}  "
             f"BB725:{variant_counts[2]}")
    cv2.putText(out, stats,
                (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    mode_text = f"Stacking: {STACKING_RULES['mode']}  |  TTA: {'ON' if ENABLE_TTA else 'OFF'}  |  FP16: {'ON' if ENABLE_FP16 else 'OFF'}"
    cv2.putText(out, mode_text,
                (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    # Panel bawah
    cv2.rectangle(out, (0, h - 32), (w, h), (30, 30, 30), -1)
    cv2.putText(out,
        "[SPACE] Capture  [R] Reset  [S] Screenshot  [T] Mode  [A] TTA  [Q] Quit",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    return out

# ============= MAIN =============
print("=" * 60)
print("PALLET VARIANT DETECTOR v2 — YOLO26 EDITION")
print("=" * 60)
print(f"\n  Model      : {MODEL_PATH}")
print(f"  Confidence : {CONFIDENCE_THRESHOLD}")
print(f"  Smoothing  : {SMOOTHING_FRAMES} frames (weighted voting)")
print(f"  TTA        : {'ON' if ENABLE_TTA else 'OFF (toggle [A])'}")
print(f"  FP16       : {'ON' if ENABLE_FP16 else 'OFF'}")
print(f"\n  Variant Config:")
for cid, cfg in VARIANT_CONFIG.items():
    print(f"    [{cid}] {cfg['name']:25s}: "
          f"{cfg['box_per_layer']} box/layer x {cfg['total_layers']} layer "
          f"= {cfg['total_boxes']} box")
print(f"\n  Stacking mode : {STACKING_RULES['mode']}")
if STACKING_RULES['mode'] == 'sequence':
    seq = [VARIANT_CONFIG.get(c, {}).get('name', '?')
           for c in STACKING_RULES['expected_sequence']]
    print(f"  Urutan        : {' -> '.join(seq)} (bawah ke atas)")
    print(f"  (Ubah expected_sequence di STACKING_RULES jika urutan berbeda)")

try:
    model = YOLO(MODEL_PATH)
    print(f"\n  Model loaded OK")
    # Auto-detect GPU untuk FP16
    try:
        import torch
        if not torch.cuda.is_available():
            ENABLE_FP16 = False
            print(f"  GPU tidak tersedia → FP16 disabled")
        else:
            print(f"  GPU: {torch.cuda.get_device_name(0)} → FP16 enabled")
    except ImportError:
        ENABLE_FP16 = False
        print(f"  PyTorch tanpa CUDA → FP16 disabled")
except Exception as e:
    print(f"\n  Gagal load model: {e}")
    exit(1)

print("\nOpening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("  Gagal buka camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"  Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("\n[SPACE] Capture | [R] Reset | [S] Screenshot | [T] Mode | [A] TTA | [Q] Quit\n")
print("=" * 60)

last_detections = []
last_valid      = False
last_detail     = "Menunggu deteksi..."
last_rows       = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    results    = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False,
                       imgsz=640, half=ENABLE_FP16, augment=ENABLE_TTA)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': float(box.conf[0]),
                'class':      int(box.cls[0])
            })

    get_smoothed_detections(detections)

    if detections:
        last_detections = detections
        last_valid, last_detail, last_rows = validate_stacking(detections)

    frame_display = draw_overlay(
        frame, last_detections, last_valid, last_detail, last_rows)
    cv2.imshow('Pallet Variant Detector v2 [YOLO26]', frame_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if last_detections:
            # Gunakan weighted voting untuk menentukan kelas dominan
            w_cls, w_conf = get_weighted_dominant_class(last_detections)
            if w_cls is not None:
                cls  = w_cls
                conf = w_conf
            else:
                main = max(last_detections, key=lambda d: d['confidence'])
                cls  = main['class']
                conf = main['confidence']
            total_detections += 1
            variant_counts[cls] = variant_counts.get(cls, 0) + 1

            log_to_excel(cls, conf, last_valid, last_detail,
                         notes=f"{len(last_detections)} det, {len(last_rows)} baris, weighted")

            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            spath = os.path.join(OUTPUT_PATH, f"detection_{ts}.jpg")
            cv2.imwrite(spath, frame_display)

            info      = VARIANT_CONFIG.get(cls, {})
            valid_str = "VALID" if last_valid else "INVALID"
            print(f"\n  CAPTURED -> {info.get('name','?')}  conf={conf:.0%}  "
                  f"status={valid_str}  total={total_detections}  [weighted]")
            print(f"  Detail: {last_detail}")
            print(f"  Pallet : {info.get('box_per_layer','?')} box/layer x "
                  f"{info.get('total_layers','?')} layer = "
                  f"{info.get('total_boxes','?')} box")
        else:
            print("\n  Tidak ada deteksi!")

    elif key in (ord('r'), ord('R')):
        total_detections = 0
        variant_counts   = {0: 0, 1: 0, 2: 0}
        detection_buffer.clear()
        last_detections  = []
        last_valid       = False
        last_detail      = "Menunggu deteksi..."
        last_rows        = []
        print("\n  Statistik di-reset")

    elif key in (ord('s'), ord('S')):
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        spath = os.path.join(OUTPUT_PATH, f"screenshot_{ts}.jpg")
        cv2.imwrite(spath, frame_display)
        print(f"\n  Screenshot: {spath}")

    elif key in (ord('t'), ord('T')):
        STACKING_RULES['mode'] = ('any' if STACKING_RULES['mode'] == 'sequence'
                                  else 'sequence')
        print(f"\n  Stacking mode -> {STACKING_RULES['mode']}")

    elif key in (ord('a'), ord('A')):
        ENABLE_TTA = not ENABLE_TTA
        print(f"\n  TTA (Test-Time Augmentation) -> {'ON' if ENABLE_TTA else 'OFF'}")
        if ENABLE_TTA:
            print("  TTA meningkatkan akurasi tapi sedikit lebih lambat.")

    elif key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("DETECTION SUMMARY")
print("="*60)
print(f"Total Captures : {total_detections}")
for cid, cnt in variant_counts.items():
    if cnt > 0:
        info = VARIANT_CONFIG.get(cid, {})
        pct  = cnt / total_detections * 100 if total_detections > 0 else 0
        print(f"  {info.get('name', f'Class {cid}'):25s}: {cnt} ({pct:.1f}%)")
print(f"\nLog Excel : {excel_file}")
print("="*60)