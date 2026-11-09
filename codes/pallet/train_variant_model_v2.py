# -*- coding: utf-8 -*-
"""
YOLO26 Training Script - Upgrade dari YOLOv8 ke YOLO26
Perubahan utama vs v2:
  - Base model: yolov8n.pt → yolo26s.pt (Small, lebih akurat)
  - YOLO26: NMS-Free, MuSGD optimizer, STAL, ProgLoss
  - Epochs 120, patience 25 (YOLO26 lebih stabil)
  - Augmentasi ditingkatkan: mosaic 0.95, mixup 0.15, erasing 0.3
  - Multi-scale training untuk variasi aspect ratio
  - Auto-resume jika training terputus
  - Simpan ringkasan hasil training ke file teks
By: Yaser Ali Husen (v2 → YOLO26 upgrade)
"""

import os
import yaml
import json
from datetime import datetime
from ultralytics import YOLO

# ============= CONFIGURATION =============
# Gunakan path absolut relatif terhadap lokasi script ini
# agar training berjalan benar dari direktori manapun
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'data.yaml'))
MODEL_BASE      = "yolo26s.pt"  # YOLO26 Small — auto-download dari Ultralytics
EPOCHS          = 120
BATCH_SIZE      = 16
IMG_SIZE        = 640
PROJECT_NAME    = "pallet_variant_detection"
EXPERIMENT_NAME = "variant_classifier_yolo26"

# Referensi konfigurasi pallet (untuk dokumentasi & validasi post-training)
PALLET_CONFIG = {
    'Sachet_20gr_LunchBox': {'box_per_layer': 14, 'total_layers': 6, 'total_boxes': 84},
    'Variant_YB77':         {'box_per_layer': 20, 'total_layers': 6, 'total_boxes': 120},
    'Variant_BB725':        {'box_per_layer': 13, 'total_layers': 7, 'total_boxes': 91},
}

# ============= AUTO DEVICE DETECT =============
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU  : {gpu_name} ({vram:.1f} GB VRAM)")
            global BATCH_SIZE
            if vram < 4:
                BATCH_SIZE = 8
                print(f"  VRAM < 4GB -> Batch size diturunkan ke {BATCH_SIZE}")
            return '0'
        else:
            print("  GPU tidak tersedia, menggunakan CPU (lebih lambat)")
            return 'cpu'
    except ImportError:
        print("  PyTorch tidak terdeteksi, menggunakan CPU")
        return 'cpu'

# ============= TRAINING =============
def train_model():
    print("=" * 60)
    print("PALLET VARIANT DETECTION - MODEL TRAINING [YOLO26]")
    print("=" * 60)

    # Tampilkan konfigurasi pallet
    print("\n  Pallet Configuration:")
    for name, cfg in PALLET_CONFIG.items():
        print(f"    {name:25s}: {cfg['box_per_layer']} box/layer x "
              f"{cfg['total_layers']} layer = {cfg['total_boxes']} box")

    # --- Validasi dataset ---
    if not os.path.exists(DATASET_PATH):
        print(f"\n  Dataset tidak ditemukan: {DATASET_PATH}")
        print("  Jalankan pallet_dataset_collector_v2.py terlebih dahulu!")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset_config = yaml.safe_load(f)

    nc    = dataset_config.get('nc', 0)
    names = dataset_config.get('names', [])
    print(f"\n  Dataset path : {dataset_config.get('path', 'N/A')}")
    print(f"  Classes      : {nc} -> {names}")

    if nc < 2:
        print("\n  Dataset harus punya minimal 2 kelas. Cek data.yaml!")
        return

    dataset_abs = dataset_config.get('path', os.path.dirname(os.path.abspath(DATASET_PATH)))
    train_dir   = os.path.join(dataset_abs, 'train', 'images')
    val_dir     = os.path.join(dataset_abs, 'val',   'images')
    n_train = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')]) if os.path.exists(train_dir) else 0
    n_val   = len([f for f in os.listdir(val_dir)   if f.endswith('.jpg')]) if os.path.exists(val_dir)   else 0

    print(f"  Train images : {n_train}")
    print(f"  Val   images : {n_val}")

    if n_train == 0:
        print("\n  Folder train kosong! Generate split dulu dengan tekan [G] di collector.")
        return

    per_class = n_train // nc
    if per_class < 50:
        print(f"\n  Peringatan: ~{per_class} gambar/kelas — idealnya minimal 150+")
        print("  Tambahkan lebih banyak foto untuk hasil lebih robust.")

    device = get_device()
    global BATCH_SIZE

    print(f"\n  Model base  : {MODEL_BASE}")
    print(f"  Epochs      : {EPOCHS} (early stop patience=25)")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Image size  : {IMG_SIZE}")
    print(f"  Device      : {device}")
    print(f"  Experiment  : {PROJECT_NAME}/{EXPERIMENT_NAME}")

    # Cek apakah ada checkpoint untuk resume
    runs_dir = os.path.join(SCRIPT_DIR, 'runs', 'detect')
    save_dir = os.path.join(runs_dir, PROJECT_NAME, EXPERIMENT_NAME)
    last_ckpt = os.path.join(save_dir, 'weights', 'last.pt')
    if os.path.exists(last_ckpt):
        print(f"\n  Checkpoint ditemukan: {last_ckpt}")
        resume = input("  Resume training? [y/N]: ").strip().lower() == 'y'
    else:
        resume = False

    base_model = last_ckpt if resume else MODEL_BASE
    print(f"\n  Base model  : {base_model}")

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    model   = YOLO(base_model)
    results = model.train(
        data       = DATASET_PATH,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        project    = os.path.join(runs_dir, PROJECT_NAME),
        name       = EXPERIMENT_NAME,
        resume     = resume,
        plots      = True,
        save       = True,
        device     = device,

        # ── Anti-overfitting (YOLO26 optimized) ──
        patience     = 25,       # YOLO26 converge lebih stabil, beri ruang lebih
        dropout      = 0.15,     # sedikit lebih tinggi untuk regularisasi
        weight_decay = 0.0005,

        # ── Augmentasi (ditingkatkan untuk YOLO26) ──
        hsv_h       = 0.03,
        hsv_s       = 0.7,
        hsv_v       = 0.5,
        degrees     = 15,
        translate   = 0.15,
        scale       = 0.8,       # ↑ dari 0.6 — variasi ukuran lebih agresif
        flipud      = 0.0,
        fliplr      = 0.4,
        mosaic      = 0.95,      # ↑ dari 0.8 — YOLO26 cocok dengan mosaic tinggi
        shear       = 5.0,
        perspective = 0.0005,
        copy_paste  = 0.15,      # ↑ dari 0.1
        mixup       = 0.15,      # ← BARU: blending gambar untuk regularisasi
        erasing     = 0.3,       # ← BARU: Random Erasing untuk robustness

        # ── Learning rate (YOLO26 tuned) ──
        lr0           = 0.01,
        lrf           = 0.01,    # ↑ dari 0.001 — YOLO26 lebih baik dgn LR akhir lebih tinggi
        warmup_epochs = 5,

        # ── Multi-scale & Misc ──
        rect     = False,        # ← BARU: disable rect training → multi-scale diversity
        verbose  = True,
        exist_ok = True,
        cache    = True,
    )

    # Dapatkan folder tempat training disimpan secara dinamis jika trainer tersedia
    if hasattr(model, 'trainer') and model.trainer is not None:
        save_dir = str(model.trainer.save_dir)
    else:
        save_dir = os.path.join('runs', 'detect', PROJECT_NAME, EXPERIMENT_NAME)

    best_model = os.path.join(save_dir, 'weights', 'best.pt')

    # ── Simpan ringkasan hasil ──
    summary = {
        'timestamp'      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment'     : f"{PROJECT_NAME}/{EXPERIMENT_NAME}",
        'base_model'     : base_model,
        'dataset'        : DATASET_PATH,
        'n_train'        : n_train,
        'n_val'          : n_val,
        'classes'        : names,
        'pallet_config'  : PALLET_CONFIG,
        'best_model'     : best_model,
        'epochs_run'     : EPOCHS,
        'final_metrics'  : {}
    }

    try:
        import pandas as pd
        csv_path = os.path.join(save_dir, 'results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
            summary['final_metrics'] = {
                'best_epoch': int(best_row['epoch']),
                'mAP50'     : round(float(best_row['metrics/mAP50(B)']),    4),
                'mAP50_95'  : round(float(best_row['metrics/mAP50-95(B)']), 4),
                'precision' : round(float(best_row['metrics/precision(B)']), 4),
                'recall'    : round(float(best_row['metrics/recall(B)']),    4),
            }
    except Exception:
        pass

    summary_path = os.path.join(save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING SELESAI!")
    print("=" * 60)
    print(f"\n  Best model  : {best_model}")
    print(f"  Summary     : {summary_path}")
    if summary['final_metrics']:
        m = summary['final_metrics']
        print(f"\n  Epoch terbaik : {m.get('best_epoch')}")
        print(f"  mAP50         : {m.get('mAP50', 'N/A')}")
        print(f"  mAP50-95      : {m.get('mAP50_95', 'N/A')}")
        print(f"  Precision     : {m.get('precision', 'N/A')}")
        print(f"  Recall        : {m.get('recall', 'N/A')}")

    print(f"\n  Class mapping:")
    for i, name in enumerate(names):
        cfg = PALLET_CONFIG.get(name, {})
        print(f"    [{i}] {name:25s} : {cfg.get('box_per_layer','?')} box/layer "
              f"x {cfg.get('total_layers','?')} layer = {cfg.get('total_boxes','?')} box")

    print(f"\nNext: Jalankan pallet_variant_detector_v2.py")
    print(f"  MODEL_PATH = '{best_model}'")
    print("=" * 60)

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nTraining dihentikan oleh user.")
        print("Checkpoint tersimpan di last.pt — bisa dilanjutkan dengan resume.")
    except Exception as e:
        print(f"\n  Error saat training: {e}")
        import traceback
        traceback.print_exc()