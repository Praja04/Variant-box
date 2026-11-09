# -*- coding: utf-8 -*-
"""
Pallet Dataset Collection System - CLEAN UI
Tampilan: kamera bersih saja. Semua info & log di console.
By: Yaser Ali Husen (Fixed)
"""

import cv2
import numpy as np
from datetime import datetime
import os
import json
import shutil

# ============= CONFIGURATION =============
VARIANT_CONFIG = {
    'JB': {
        'total_boxes': 12,
        'name': 'Variant JB (3x4)',
        'grid': {'rows': 3, 'cols': 4},
        'class_id': 0,
        'color': (0, 255, 0)
    },
    'BB': {
        'total_boxes': 20,
        'name': 'Variant BB (4x5)',
        'grid': {'rows': 4, 'cols': 5},
        'class_id': 1,
        'color': (255, 0, 0)
    },
    'YB': {
        'total_boxes': 14,
        'name': 'Variant YB (2x7)',
        'grid': {'rows': 2, 'cols': 7},
        'class_id': 2,
        'color': (0, 165, 255)
    },
    'NEGATIVE': {
        'name': 'No Pallet (Background)',
        'class_id': -1,
        'color': (128, 128, 128)
    }
}

USB_CAMERA_INDEX = 1

DATASET_ROOT = "../dataset/"
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
LABELS_DIR   = os.path.join(DATASET_ROOT, "labels")
RAW_DIR      = os.path.join(DATASET_ROOT, "raw")

for variant in ['JB', 'BB', 'YB', 'NEGATIVE']:
    os.makedirs(os.path.join(IMAGES_DIR, f"variant_{variant}"), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, f"variant_{variant}"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR,    f"variant_{variant}"), exist_ok=True)

METADATA_FILE = os.path.join(DATASET_ROOT, "dataset_metadata.json")

CURRENT_VARIANT = 'JB'

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

dataset_stats = {
    'JB':       {'total': 0, 'modes': {m: 0 for m in CAPTURE_MODES}},
    'BB':       {'total': 0, 'modes': {m: 0 for m in CAPTURE_MODES}},
    'YB':       {'total': 0, 'modes': {m: 0 for m in CAPTURE_MODES}},
    'NEGATIVE': {'total': 0, 'modes': {}},
    'total':    0
}

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
        dataset_stats = metadata.get('stats', dataset_stats)
else:
    metadata = {
        'created':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'variants': {k: {'name': v['name'], 'class_id': v['class_id']}
                     for k, v in VARIANT_CONFIG.items() if k != 'NEGATIVE'},
        'stats':    dataset_stats,
        'captures': []
    }

def save_metadata():
    metadata['stats'] = dataset_stats
    metadata['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def create_yolo_label(variant, image_width, image_height):
    if variant == 'NEGATIVE':
        return ''
    class_id = VARIANT_CONFIG[variant]['class_id']
    return f"{class_id} 0.500000 0.500000 0.850000 0.850000\n"

def capture_dataset_image(frame, variant, mode):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename   = f"{variant}_{mode}_{timestamp}"
    height, width = frame.shape[:2]

    raw_path   = os.path.join(RAW_DIR,    f"variant_{variant}", f"{filename}.jpg")
    img_path   = os.path.join(IMAGES_DIR, f"variant_{variant}", f"{filename}.jpg")
    label_path = os.path.join(LABELS_DIR, f"variant_{variant}", f"{filename}.txt")

    cv2.imwrite(raw_path,  frame)
    cv2.imwrite(img_path,  frame)

    with open(label_path, 'w') as f:
        f.write(create_yolo_label(variant, width, height))

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
        'image_size': {'width': width, 'height': height}
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

    for variant in ['JB', 'BB', 'YB', 'NEGATIVE']:
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

    yaml_content = f"""# Pallet Variant Detection Dataset
path: {os.path.abspath(DATASET_ROOT)}
train: train/images
val:   val/images

nc: 3
names: ['Variant_JB', 'Variant_BB', 'Variant_YB']

created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
description: JB(3x4=12 box), BB(4x5=20 box), YB(2x7=14 box)
"""
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nTrain/Val Split ({int(train_ratio*100)}/{int((1-train_ratio)*100)}):")
    for split in ['train', 'val']:
        total = sum(split_stats[split].values())
        print(f"\n  {split.upper()}:")
        for v in ['JB', 'BB', 'YB', 'NEGATIVE']:
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

def print_status():
    jb  = dataset_stats['JB']['total']
    bb  = dataset_stats['BB']['total']
    yb  = dataset_stats['YB']['total']
    neg = dataset_stats['NEGATIVE']['total']
    tot = dataset_stats['total']
    print(f"  Status  JB:{jb}  BB:{bb}  YB:{yb}  NEG:{neg}  TOTAL:{tot}")

# ============= MAIN =============
print("=" * 60)
print("PALLET DATASET COLLECTION")
print("=" * 60)
print(f"\nDataset : {os.path.abspath(DATASET_ROOT)}")
print()
print_status()
print()
print("─── KONTROL ───────────────────────────────────────")
print("  [SPACE]  Capture gambar")
print("  [J] JB   [B] BB   [Y] YB   [N] Negative")
print("  [1] center       [2] close        [3] far")
print("  [4] angle_left   [5] angle_right  [6] angle_top")
print("  [7] with_bg      [8] low_light    [9] high_light")
print("  [G] Generate train/val split")
print("  [D] Hapus capture terakhir")
print("  [Q] Keluar")
print("───────────────────────────────────────────────────")
print(f"\nVariant: {CURRENT_VARIANT}  |  Mode: {CURRENT_MODE}")
print("\nOpening camera...")

cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Gagal buka camera index {USB_CAMERA_INDEX}")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {actual_w}x{actual_h} @ {int(cap.get(cv2.CAP_PROP_FPS))}fps")
print("─── SIAP ─────────────────────────────────────────\n")

window_name = 'Camera'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

mode_keys = {
    ord('1'): 'center',          ord('2'): 'close',
    ord('3'): 'far',             ord('4'): 'angle_left',
    ord('5'): 'angle_right',     ord('6'): 'angle_top',
    ord('7'): 'with_background', ord('8'): 'low_light',
    ord('9'): 'high_light'
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal baca frame.")
        break

    # ── Tampilkan kamera bersih, tanpa overlay ──
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = capture_dataset_image(frame, CURRENT_VARIANT, CURRENT_MODE)
        print(f"[CAPTURE] {CURRENT_VARIANT} | {CURRENT_MODE} | "
              f"total {CURRENT_VARIANT}: {dataset_stats[CURRENT_VARIANT]['total']}")

    elif key in (ord('j'), ord('J')):
        CURRENT_VARIANT = 'JB'
        print(f"[VARIANT] JB  (total: {dataset_stats['JB']['total']})")

    elif key in (ord('b'), ord('B')):
        CURRENT_VARIANT = 'BB'
        print(f"[VARIANT] BB  (total: {dataset_stats['BB']['total']})")

    elif key in (ord('y'), ord('Y')):
        CURRENT_VARIANT = 'YB'
        print(f"[VARIANT] YB  (total: {dataset_stats['YB']['total']})")

    elif key in (ord('n'), ord('N')):
        CURRENT_VARIANT = 'NEGATIVE'
        print(f"[VARIANT] NEGATIVE  (total: {dataset_stats['NEGATIVE']['total']})")

    elif key in mode_keys:
        CURRENT_MODE = mode_keys[key]
        print(f"[MODE] {CAPTURE_MODES[CURRENT_MODE]}")

    elif key in (ord('g'), ord('G')):
        cap.release()
        cv2.destroyAllWindows()
        generate_train_val_split()
        cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Camera kembali aktif.\n")

    elif key in (ord('d'), ord('D')):
        print("[DELETE] Menghapus capture terakhir...")
        delete_last_capture()
        print_status()

    elif key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for var in ['JB', 'BB', 'YB', 'NEGATIVE']:
    print(f"  {var}: {dataset_stats[var]['total']} images")
print(f"  Total: {dataset_stats['total']} images")
print(f"\nDataset: {os.path.abspath(DATASET_ROOT)}")
print("Next: Tekan [G] generate split → train_variant_model.py")
print("="*60)