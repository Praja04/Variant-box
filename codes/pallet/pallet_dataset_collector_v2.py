# -*- coding: utf-8 -*-
"""
Pallet Dataset Collection System - v2 ACCURATE LABELING
Fix utama:
  - Bounding box mengikuti ROI yang digambar user (bukan hardcoded)
  - ROI selector interaktif: drag mouse untuk tentukan area kardus
  - Auto-ROI: deteksi tepi + contour sederhana sebagai bantuan
  - Semua fix lain dari v1 tetap ada
By: Yaser Ali Husen (v2 - Accurate Labeling)
"""

import cv2
import numpy as np
from datetime import datetime
import os
import json
import shutil

# ============= CONFIGURATION =============
VARIANT_CONFIG = {
    'SACHET_20GR': {
        'name':          'Sachet 20gr Lunch Box',
        'class_id':      0,
        'color':         (0, 200, 80),
        'box_per_layer': 14,
        'total_layers':  6,
        'total_boxes':   14 * 6,   # 84 box per pallet
    },
    'YB_77': {
        'name':          'Variant YB 77',
        'class_id':      1,
        'color':         (0, 165, 255),
        'box_per_layer': 20,
        'total_layers':  6,
        'total_boxes':   20 * 6,   # 120 box per pallet
    },
    'BB_725': {
        'name':          'Variant BB 725',
        'class_id':      2,
        'color':         (80, 80, 255),
        'box_per_layer': 13,
        'total_layers':  7,
        'total_boxes':   13 * 7,   # 91 box per pallet
    },
    'NEGATIVE': {
        'name':          'No Pallet (Background)',
        'class_id':      -1,
        'color':         (128, 128, 128),
        'box_per_layer': 0,
        'total_layers':  0,
        'total_boxes':   0,
    }
}

# Kamera RTSP Conveyor (sama dengan conveyor_counting)
CAMERA_SOURCE = "rtsp://admin:1q2w3e4r@10.11.9.80:554/Streaming/Channels/101"
# Jika ingin menggunakan USB webcam lokal kembali, uncomment baris di bawah:
# CAMERA_SOURCE = 0

DATASET_ROOT = "../dataset/"
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
LABELS_DIR   = os.path.join(DATASET_ROOT, "labels")
RAW_DIR      = os.path.join(DATASET_ROOT, "raw")

for variant in ['SACHET_20GR', 'YB_77', 'BB_725', 'NEGATIVE']:
    os.makedirs(os.path.join(IMAGES_DIR, f"variant_{variant}"), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, f"variant_{variant}"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR,    f"variant_{variant}"), exist_ok=True)

METADATA_FILE = os.path.join(DATASET_ROOT, "dataset_metadata.json")

CURRENT_VARIANT = 'SACHET_20GR'

CAPTURE_MODES = {
    'center':          'Center View - Normal',
    'close':           'Close Up - Dekat',
    'far':             'Far View - Jauh',
    'angle_left':      'Angle Left - Dari Kiri',
    'angle_right':     'Angle Right - Dari Kanan',
    'angle_top':       'Top View - Dari Atas',
    'with_background': 'With Background - Ada Orang/Objek',
    'low_light':       'Low Light - Cahaya Gelap',
    'high_light':      'High Light - Cahaya Terang'
}
CURRENT_MODE = 'center'

# ============= LABELING MODE =============
# 'roi'     : user drag ROI di frame sebelum capture (paling akurat)
# 'auto'    : sistem deteksi contour terbesar otomatis
# 'guide'   : tampilkan guide box di tengah, user posisikan kardus ke guide
LABEL_MODE = 'roi'

# ============= ROI STATE =============
roi_rect      = None   # (x1, y1, x2, y2) pixel coords
drawing       = False
roi_start     = None
roi_confirmed = False

dataset_stats = {
    'SACHET_20GR': {'total': 0, 'modes': {m: 0 for m in CAPTURE_MODES}},
    'YB_77':       {'total': 0, 'modes': {m: 0 for m in CAPTURE_MODES}},
    'BB_725':      {'total': 0, 'modes': {m: 0 for m in CAPTURE_MODES}},
    'NEGATIVE':    {'total': 0, 'modes': {}},
    'total':       0
}

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
        dataset_stats = metadata.get('stats', dataset_stats)
else:
    metadata = {
        'created':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'variants': {
            k: {
                'name':          v['name'],
                'class_id':      v['class_id'],
                'box_per_layer': v['box_per_layer'],
                'total_layers':  v['total_layers'],
                'total_boxes':   v['total_boxes'],
            }
            for k, v in VARIANT_CONFIG.items() if k != 'NEGATIVE'
        },
        'stats':    dataset_stats,
        'captures': []
    }

def save_metadata():
    metadata['stats'] = dataset_stats
    metadata['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

# ============= YOLO LABEL - AKURAT =============
def create_yolo_label_from_roi(variant, roi, image_width, image_height):
    """
    Buat label YOLO dari ROI yang sebenarnya.
    roi = (x1, y1, x2, y2) dalam pixel
    Output: "class_id cx cy w h" dalam format normalized [0..1]
    """
    if variant == 'NEGATIVE':
        return ''
    if roi is None:
        return f"{VARIANT_CONFIG[variant]['class_id']} 0.500000 0.500000 1.000000 1.000000\n"

    class_id = VARIANT_CONFIG[variant]['class_id']
    x1, y1, x2, y2 = roi

    x1 = max(0, min(x1, image_width  - 1))
    x2 = max(0, min(x2, image_width  - 1))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height - 1))

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    cx = (x1 + x2) / 2.0 / image_width
    cy = (y1 + y2) / 2.0 / image_height
    w  = (x2 - x1) / float(image_width)
    h  = (y2 - y1) / float(image_height)

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

# ============= VALIDASI LAYER =============
def validate_layer_count(variant, detected_boxes):
    """
    Validasi jumlah box terdeteksi terhadap konfigurasi pallet.
    detected_boxes : jumlah box yang terdeteksi di frame
    Return: dict berisi info validasi
    """
    cfg = VARIANT_CONFIG.get(variant)
    if not cfg or variant == 'NEGATIVE':
        return None

    box_per_layer  = cfg['box_per_layer']
    total_layers   = cfg['total_layers']
    total_expected = cfg['total_boxes']

    if box_per_layer == 0:
        return None

    estimated_layers = round(detected_boxes / box_per_layer)
    is_valid_layer   = (detected_boxes % box_per_layer == 0)
    is_full_pallet   = (detected_boxes == total_expected)

    return {
        'variant':          variant,
        'detected_boxes':   detected_boxes,
        'box_per_layer':    box_per_layer,
        'total_layers':     total_layers,
        'total_expected':   total_expected,
        'estimated_layers': estimated_layers,
        'is_valid_layer':   is_valid_layer,
        'is_full_pallet':   is_full_pallet,
        'status': (
            'FULL PALLET'       if is_full_pallet else
            'LAYER VALID'       if is_valid_layer else
            'LAYER TIDAK VALID'
        )
    }

# ============= AUTO ROI: deteksi contour terbesar =============
def auto_detect_roi(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    h, w = frame.shape[:2]
    frame_area = w * h
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.05:
            continue
        if area > frame_area * 0.98:
            continue
        if area > best_area:
            best_area = area
            best = cnt

    if best is None:
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return (x1, y1, x2, y2)

# ============= MOUSE CALLBACK untuk ROI drag =============
def mouse_callback(event, x, y, flags, param):
    global roi_rect, drawing, roi_start, roi_confirmed

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing       = True
        roi_confirmed = False
        roi_start     = (x, y)
        roi_rect      = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_rect = (roi_start[0], roi_start[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing  = False
        roi_rect = (roi_start[0], roi_start[1], x, y)
        x1, y1, x2, y2 = roi_rect
        if abs(x2 - x1) > 50 and abs(y2 - y1) > 50:
            roi_confirmed = True
            print(f"  [ROI] Ditentukan: ({x1},{y1}) → ({x2},{y2})  "
                  f"size={abs(x2-x1)}x{abs(y2-y1)}px")
        else:
            roi_rect = None
            print("  [ROI] Terlalu kecil, coba lagi")

def draw_frame_overlay(frame, variant, mode, label_mode, current_roi, auto_roi):
    out   = frame.copy()
    h, w  = out.shape[:2]
    cfg   = VARIANT_CONFIG[variant]
    color = cfg['color']

    # ── ROI yang sedang digambar ──
    if current_roi:
        x1, y1, x2, y2 = current_roi
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if roi_confirmed:
            cv2.putText(out, "ROI CONFIRMED", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── Auto ROI suggestion ──
    if auto_roi and label_mode == 'auto':
        ax1, ay1, ax2, ay2 = auto_roi
        cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (200, 200, 200), 1)
        cv2.putText(out, "auto-detect", (ax1, ay1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── Guide box mode ──
    if label_mode == 'guide':
        gx1 = int(w * 0.1); gy1 = int(h * 0.1)
        gx2 = int(w * 0.9); gy2 = int(h * 0.9)
        cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
        cv2.putText(out, "Posisikan kardus dalam kotak ini",
                    (gx1 + 5, gy1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── Status bar atas ──
    cv2.rectangle(out, (0, 0), (w, 60), (0, 0, 0), -1)
    status_roi = "ROI: SIAP v" if roi_confirmed else "ROI: Belum (drag mouse)"
    if label_mode == 'auto':   status_roi = "Mode: AUTO DETECT"
    elif label_mode == 'guide': status_roi = "Mode: GUIDE BOX"

    pallet_info = ""
    if variant != 'NEGATIVE':
        pallet_info = (f"{cfg['box_per_layer']} box/layer x "
                       f"{cfg['total_layers']} layer = {cfg['total_boxes']} box total")

    cv2.putText(out,
        f"Variant: {variant}  |  Mode: {mode[:10]}  |  {status_roi}",
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(out, pallet_info,
        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(out,
        f"SACHET:{dataset_stats['SACHET_20GR']['total']}  "
        f"YB77:{dataset_stats['YB_77']['total']}  "
        f"BB725:{dataset_stats['BB_725']['total']}  "
        f"NEG:{dataset_stats['NEGATIVE']['total']}",
        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # ── Hint bawah ──
    cv2.rectangle(out, (0, h - 30), (w, h), (30, 30, 30), -1)
    if label_mode == 'roi':
        hint = "[DRAG] ROI  [SPACE] Capture  [C] Clear  [A] Auto-detect  [Q] Quit"
    elif label_mode == 'auto':
        hint = "[SPACE] Capture auto-ROI  [L] Ganti mode  [Q] Quit"
    else:
        hint = "[SPACE] Capture guide box  [L] Ganti mode  [Q] Quit"
    cv2.putText(out, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return out

def capture_dataset_image(frame, variant, mode, roi):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename   = f"{variant}_{mode}_{timestamp}"
    height, width = frame.shape[:2]

    raw_path   = os.path.join(RAW_DIR,    f"variant_{variant}", f"{filename}.jpg")
    img_path   = os.path.join(IMAGES_DIR, f"variant_{variant}", f"{filename}.jpg")
    label_path = os.path.join(LABELS_DIR, f"variant_{variant}", f"{filename}.txt")

    cv2.imwrite(raw_path, frame)
    cv2.imwrite(img_path, frame)

    label_str = create_yolo_label_from_roi(variant, roi, width, height)
    with open(label_path, 'w') as f:
        f.write(label_str)

    dataset_stats[variant]['total'] += 1
    if variant != 'NEGATIVE':
        dataset_stats[variant]['modes'][mode] = \
            dataset_stats[variant]['modes'].get(mode, 0) + 1
    dataset_stats['total'] += 1

    metadata['captures'].append({
        'timestamp':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'variant':    variant,
        'mode':       mode,
        'filename':   filename,
        'image_path': img_path,
        'label_path': label_path,
        'image_size': {'width': width, 'height': height},
        'roi':        list(roi) if roi else None,
        'label':      label_str.strip()
    })
    save_metadata()
    return filename

def generate_train_val_split(train_ratio=0.8):
    print("\n" + "="*60)
    print("GENERATING TRAIN/VAL SPLIT")
    print("="*60)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, split, 'labels'), exist_ok=True)

    split_stats = {'train': {}, 'val': {}}

    for variant in ['SACHET_20GR', 'YB_77', 'BB_725', 'NEGATIVE']:
        img_dir = os.path.join(IMAGES_DIR, f"variant_{variant}")
        lbl_dir = os.path.join(LABELS_DIR, f"variant_{variant}")

        if not os.path.exists(img_dir):
            continue

        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        if not images:
            continue

        np.random.shuffle(images)
        n_train = max(1, int(len(images) * train_ratio))
        splits  = {'train': images[:n_train], 'val': images[n_train:]}

        for split, imgs in splits.items():
            split_stats[split][variant] = len(imgs)
            for img_name in imgs:
                base = os.path.splitext(img_name)[0]
                shutil.copy(os.path.join(img_dir, img_name),
                            os.path.join(DATASET_ROOT, split, 'images', img_name))
                src_lbl = os.path.join(lbl_dir, f"{base}.txt")
                if os.path.exists(src_lbl):
                    shutil.copy(src_lbl,
                                os.path.join(DATASET_ROOT, split, 'labels', f"{base}.txt"))

    yaml_content = f"""# Pallet Variant Detection Dataset - v2 Accurate Labels
path: {os.path.abspath(DATASET_ROOT)}
train: train/images
val:   val/images

nc: 3
names: ['Sachet_20gr_LunchBox', 'Variant_YB77', 'Variant_BB725']

# Pallet configuration (box_per_layer x total_layers = total_boxes per pallet)
# Sachet_20gr_LunchBox : 14 box/layer x 6 layer = 84 box
# Variant_YB77         : 20 box/layer x 6 layer = 120 box
# Variant_BB725        : 13 box/layer x 7 layer = 91 box

created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nTrain/Val Split ({int(train_ratio*100)}/{int((1-train_ratio)*100)}):")
    for split in ['train', 'val']:
        total = sum(split_stats[split].values())
        print(f"\n  {split.upper()}:")
        for v in ['SACHET_20GR', 'YB_77', 'BB_725', 'NEGATIVE']:
            c = split_stats[split].get(v, 0)
            if c > 0:
                print(f"    {v}: {c} images")
        print(f"    Total: {total} images")
    print(f"\n  data.yaml: {yaml_path}")
    print("="*60)

def delete_last_capture():
    if not metadata['captures']:
        print("  Tidak ada capture untuk dihapus.")
        return False
    last    = metadata['captures'][-1]
    variant = last['variant']
    mode    = last.get('mode', 'center')
    try:
        if os.path.exists(last['image_path']):
            os.remove(last['image_path'])
        if os.path.exists(last['label_path']):
            os.remove(last['label_path'])
        dataset_stats[variant]['total'] -= 1
        if variant != 'NEGATIVE':
            dataset_stats[variant]['modes'][mode] = \
                max(0, dataset_stats[variant]['modes'].get(mode, 0) - 1)
        dataset_stats['total'] -= 1
        metadata['captures'].pop()
        save_metadata()
        print(f"  Deleted: {last['filename']}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def validate_existing_labels():
    print("\n" + "="*60)
    print("VALIDASI LABEL & KONFIGURASI PALLET")
    print("="*60)

    print("\n  Konfigurasi Variant:")
    for vk, vc in VARIANT_CONFIG.items():
        if vk == 'NEGATIVE': continue
        print(f"    {vk:15s} : {vc['box_per_layer']} box/layer x "
              f"{vc['total_layers']} layer = {vc['total_boxes']} box total")

    print("\n  Cek Label Files:")
    hardcoded_count = 0
    total_labels    = 0
    for variant in ['SACHET_20GR', 'YB_77', 'BB_725']:
        lbl_dir = os.path.join(LABELS_DIR, f"variant_{variant}")
        if not os.path.exists(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            total_labels += 1
            fpath = os.path.join(lbl_dir, fname)
            with open(fpath) as f:
                content = f.read().strip()
            parts = content.split()
            if len(parts) == 5:
                cx, cy = float(parts[1]), float(parts[2])
                if abs(cx - 0.5) < 0.01 and abs(cy - 0.5) < 0.01:
                    hardcoded_count += 1

    print(f"    Total label files : {total_labels}")
    print(f"    Hardcoded (lama)  : {hardcoded_count} — perlu diambil ulang")
    print(f"    Akurat (ROI)      : {total_labels - hardcoded_count}")
    if hardcoded_count > 0:
        print(f"\n    Saran: ambil ulang {hardcoded_count} gambar dengan versi ini.")
    print("="*60)

# ============= MAIN =============
print("=" * 60)
print("PALLET DATASET COLLECTION v2 - ACCURATE LABELING")
print("=" * 60)
print(f"\nDataset   : {os.path.abspath(DATASET_ROOT)}")
print(f"Label mode: {LABEL_MODE}")
print()
print("  Variant Config:")
for vk, vc in VARIANT_CONFIG.items():
    if vk == 'NEGATIVE': continue
    print(f"    [{vc['class_id']}] {vk:15s} : {vc['box_per_layer']} box/layer "
          f"x {vc['total_layers']} layer = {vc['total_boxes']} box")
print()
validate_existing_labels()
print()
print("─── KONTROL ──────────────────────────────────────────")
print("  [DRAG MOUSE]  Gambar ROI di sekitar kardus")
print("  [SPACE]       Capture")
print("  [A]           Auto-detect ROI dari contour")
print("  [C]           Clear / reset ROI")
print("  [L]           Toggle label mode (roi/auto/guide)")
print("  [S] SACHET_20GR   [Y] YB_77   [B] BB_725   [N] Negative")
print("  [1]-[9]       Ganti capture mode")
print("  [G]           Generate train/val split")
print("  [D]           Hapus capture terakhir")
print("  [V]           Validasi label yang ada")
print("  [Q]           Keluar")
print("──────────────────────────────────────────────────────")
print(f"\nVariant: {CURRENT_VARIANT}  |  Mode capture: {CURRENT_MODE}")
print("\nOpening camera...")

is_rtsp = isinstance(CAMERA_SOURCE, str) and ("rtsp" in CAMERA_SOURCE or "http" in CAMERA_SOURCE)

if is_rtsp:
    cap = cv2.VideoCapture(CAMERA_SOURCE)
else:
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"  Gagal buka camera source: {CAMERA_SOURCE}")
    if is_rtsp:
        exit(1)
    else:
        print("  Mencoba camera index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("  Gagal buka camera!")
            exit(1)

if not is_rtsp:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if not is_rtsp else 1920
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if not is_rtsp else 1080
print(f"Camera: {actual_w}x{actual_h} @ {int(cap.get(cv2.CAP_PROP_FPS))}fps")

window_name = 'Pallet Collector v2'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_callback)

mode_keys = {
    ord('1'): 'center',          ord('2'): 'close',
    ord('3'): 'far',             ord('4'): 'angle_left',
    ord('5'): 'angle_right',     ord('6'): 'angle_top',
    ord('7'): 'with_background', ord('8'): 'low_light',
    ord('9'): 'high_light'
}

label_modes    = ['roi', 'auto', 'guide']
label_mode_idx = 0

print("─── SIAP ─────────────────────────────────────────────\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal baca frame.")
        break

    auto_roi = None
    if LABEL_MODE == 'auto':
        auto_roi = auto_detect_roi(frame)

    frame_display = draw_frame_overlay(
        frame, CURRENT_VARIANT, CURRENT_MODE, LABEL_MODE, roi_rect, auto_roi)
    cv2.imshow(window_name, frame_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if LABEL_MODE == 'roi':
            if not roi_confirmed or roi_rect is None:
                print("[CAPTURE] x Gambar ROI dulu (drag mouse di sekitar kardus)!")
                continue
            capture_roi = roi_rect
        elif LABEL_MODE == 'auto':
            capture_roi = auto_detect_roi(frame)
            if capture_roi is None:
                print("[CAPTURE] x Auto-detect gagal, coba mode ROI manual (tekan L)")
                continue
            print(f"  [AUTO-ROI] {capture_roi}")
        else:  # guide
            h_frame, w_frame = frame.shape[:2]
            capture_roi = (int(w_frame*0.1), int(h_frame*0.1),
                           int(w_frame*0.9), int(h_frame*0.9))

        filename = capture_dataset_image(frame, CURRENT_VARIANT, CURRENT_MODE, capture_roi)
        label    = create_yolo_label_from_roi(
            CURRENT_VARIANT, capture_roi, frame.shape[1], frame.shape[0]).strip()
        cfg = VARIANT_CONFIG[CURRENT_VARIANT]
        print(f"[CAPTURE] {CURRENT_VARIANT} | {CURRENT_MODE} | ROI={capture_roi} | "
              f"label='{label}' | total: {dataset_stats[CURRENT_VARIANT]['total']}  "
              f"({cfg.get('box_per_layer',0)} box/layer x {cfg.get('total_layers',0)} layer)")

    elif key in (ord('a'), ord('A')):
        detected = auto_detect_roi(frame)
        if detected:
            roi_rect      = detected
            roi_confirmed = True
            print(f"  [AUTO] ROI otomatis: {detected}")
        else:
            print("  [AUTO] Tidak ada objek terdeteksi")

    elif key in (ord('c'), ord('C')):
        roi_rect      = None
        roi_confirmed = False
        print("  [ROI] Cleared")

    elif key in (ord('l'), ord('L')):
        label_mode_idx = (label_mode_idx + 1) % len(label_modes)
        LABEL_MODE     = label_modes[label_mode_idx]
        roi_rect       = None
        roi_confirmed  = False
        print(f"  [MODE] Label mode -> {LABEL_MODE}")

    elif key in (ord('s'), ord('S')):
        CURRENT_VARIANT = 'SACHET_20GR'
        roi_rect = None; roi_confirmed = False
        print(f"[VARIANT] SACHET_20GR  (total: {dataset_stats['SACHET_20GR']['total']})")

    elif key in (ord('y'), ord('Y')):
        CURRENT_VARIANT = 'YB_77'
        roi_rect = None; roi_confirmed = False
        print(f"[VARIANT] YB_77  (total: {dataset_stats['YB_77']['total']})")

    elif key in (ord('b'), ord('B')):
        CURRENT_VARIANT = 'BB_725'
        roi_rect = None; roi_confirmed = False
        print(f"[VARIANT] BB_725  (total: {dataset_stats['BB_725']['total']})")

    elif key in (ord('n'), ord('N')):
        CURRENT_VARIANT = 'NEGATIVE'
        roi_rect = None; roi_confirmed = False
        print(f"[VARIANT] NEGATIVE  (total: {dataset_stats['NEGATIVE']['total']})")

    elif key in mode_keys:
        CURRENT_MODE = mode_keys[key]
        print(f"[MODE] {CAPTURE_MODES[CURRENT_MODE]}")

    elif key in (ord('g'), ord('G')):
        cap.release()
        cv2.destroyAllWindows()
        generate_train_val_split()
        if is_rtsp:
            cap = cv2.VideoCapture(CAMERA_SOURCE)
        else:
            cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        print("Camera kembali aktif.\n")

    elif key in (ord('d'), ord('D')):
        print("[DELETE] Menghapus capture terakhir...")
        delete_last_capture()

    elif key in (ord('v'), ord('V')):
        validate_existing_labels()

    elif key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for var in ['SACHET_20GR', 'YB_77', 'BB_725', 'NEGATIVE']:
    cfg   = VARIANT_CONFIG[var]
    extra = ""
    if var != 'NEGATIVE':
        extra = f"  ({cfg['box_per_layer']} box/layer x {cfg['total_layers']} layer = {cfg['total_boxes']} box)"
    print(f"  {var:15s}: {dataset_stats[var]['total']} images{extra}")
print(f"  {'Total':15s}: {dataset_stats['total']} images")
print(f"\nDataset: {os.path.abspath(DATASET_ROOT)}")
print("Next: Tekan [G] generate split -> train_variant_model_v2.py")
print("="*60)