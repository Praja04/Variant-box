# -*- coding: utf-8 -*-
"""
YOLO Training Script for Pallet Variant Detection - FIXED
Fix: Auto GPU detect, dataset path validation, augmentasi lebih baik
"""

import os
import yaml
from ultralytics import YOLO

# ============= CONFIGURATION =============
DATASET_PATH   = "../dataset/data.yaml"
MODEL_BASE     = "yolov8n.pt"       # Nano = cepat. Ganti yolov8s.pt jika akurasi kurang
EPOCHS         = 50
BATCH_SIZE     = 16
IMG_SIZE       = 640
PROJECT_NAME   = "pallet_variant_detection"
EXPERIMENT_NAME = "variant_classifier_v1"

# ============= AUTO DEVICE DETECT =============
def get_device():
    """FIX: Otomatis pakai GPU jika tersedia"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU ditemukan: {gpu_name}")
            return '0'       # GPU pertama
        else:
            print("⚠ GPU tidak tersedia, menggunakan CPU (lebih lambat)")
            return 'cpu'
    except ImportError:
        print("⚠ PyTorch tidak terdeteksi, menggunakan CPU")
        return 'cpu'

# ============= TRAINING =============
def train_model():
    print("=" * 60)
    print("PALLET VARIANT DETECTION - MODEL TRAINING")
    print("=" * 60)

    # --- Validasi dataset ---
    if not os.path.exists(DATASET_PATH):
        print(f"\n✗ Dataset tidak ditemukan: {DATASET_PATH}")
        print("\nLangkah yang harus dilakukan dulu:")
        print("  1. Jalankan: python pallet_dataset_collector.py")
        print("  2. Kumpulkan minimal 100 foto per variant (JB, BB, YB)")
        print("  3. Tekan [G] di program collector untuk generate train/val split")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # Validasi isi data.yaml
    nc    = dataset_config.get('nc', 0)
    names = dataset_config.get('names', [])
    print(f"\nDataset Configuration:")
    print(f"  Path   : {dataset_config.get('path', 'N/A')}")
    print(f"  Classes: {nc}")
    print(f"  Names  : {names}")

    if nc != 3 or len(names) != 3:
        print(f"\n✗ Dataset harus punya 3 kelas (JB, BB, YB). Cek data.yaml!")
        return

    # Cek folder train/val ada isinya
    dataset_abs = dataset_config.get('path', os.path.dirname(os.path.abspath(DATASET_PATH)))
    train_dir   = os.path.join(dataset_abs, 'train', 'images')
    val_dir     = os.path.join(dataset_abs, 'val',   'images')
    n_train = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')]) if os.path.exists(train_dir) else 0
    n_val   = len([f for f in os.listdir(val_dir)   if f.endswith('.jpg')]) if os.path.exists(val_dir)   else 0

    print(f"\n  Train images: {n_train}")
    print(f"  Val   images: {n_val}")

    if n_train == 0:
        print("\n✗ Folder train kosong! Generate split dulu dengan tekan [G] di collector.")
        return

    device = get_device()

    print(f"\nTraining Configuration:")
    print(f"  Base Model : {MODEL_BASE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch Size : {BATCH_SIZE}")
    print(f"  Image Size : {IMG_SIZE}")
    print(f"  Device     : {device}")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    
    model = YOLO('pallet_variant_detection/variant_classifier_v1/weights/best.pt')
    results = model.train(
        data       = DATASET_PATH,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        project    = PROJECT_NAME,
        name       = EXPERIMENT_NAME,
        plots      = True,
        save       = True,
        device     = device,   # FIX: auto GPU/CPU

        # Augmentasi (sesuaikan jika dataset sudah beragam)
        hsv_h      = 0.05,
        hsv_s      = 0.9,
        hsv_v      = 0.6,
        degrees    = 30,
        translate  = 0.2,
        scale      = 0.7,
        flipud     = 0.0,
        fliplr     = 0.5,
        mosaic     = 1.0,
        shear      = 10.0,
        perspective= 0.001,

        # Learning rate
        lr0        = 0.01,
        lrf        = 0.01,

        # Early stopping (hentikan training jika tidak ada improvement)
        patience   = 30,
        verbose    = True,
    )

    best_model = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, 'weights', 'best.pt')

    print("\n" + "=" * 60)
    print("✅ TRAINING SELESAI!")
    print("=" * 60)
    print(f"\nModel terbaik: {best_model}")
    print(f"Model terakhir: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/last.pt")
    print(f"Training plots: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
    print("\nNext: Jalankan pallet_variant_detector.py")
    print(f"  Pastikan MODEL_PATH = '{best_model}'")
    print("=" * 60)

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nTraining dihentikan oleh user.")
    except Exception as e:
        print(f"\n✗ Error saat training: {e}")
        import traceback
        traceback.print_exc()