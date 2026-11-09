# -*- coding: utf-8 -*-
"""
Reset Dataset and Model
Hapus semua data lama untuk mulai dari awal
"""

import os
import shutil

print("=" * 60)
print("RESET DATASET AND MODEL")
print("=" * 60)
print("\nScript ini akan menghapus:")
print("  1. Dataset folder (images, labels, train, val)")
print("  2. Model training results")
print("  3. Output logs")
print("  4. Metadata JSON")
print("\n⚠️  PERINGATAN: Data akan DIHAPUS PERMANEN!")
print("=" * 60)

# Konfirmasi
confirm = input("\nKetik 'HAPUS' untuk konfirmasi: ")

if confirm != "HAPUS":
    print("\n❌ Dibatalkan. Data tidak dihapus.")
    exit(0)

print("\n🗑️  Menghapus data lama...\n")

# Folders to delete
folders_to_delete = [
    "../dataset/",
    "../output/",
    "pallet_variant_detection/",
    "runs/",
]

files_to_delete = [
    "dataset_metadata.json",
]

# Delete folders
for folder in folders_to_delete:
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f"✓ Deleted: {folder}")
        except Exception as e:
            print(f"✗ Error deleting {folder}: {e}")
    else:
        print(f"⊘ Not found: {folder}")

# Delete files
for file in files_to_delete:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"✓ Deleted: {file}")
        except Exception as e:
            print(f"✗ Error deleting {file}: {e}")
    else:
        print(f"⊘ Not found: {file}")

print("\n" + "=" * 60)
print("✅ RESET SELESAI!")
print("=" * 60)
print("\nSemua data lama sudah dihapus.")
print("\nSelanjutnya:")
print("  1. Jalankan: python pallet_dataset_collector_v2.py")
print("  2. Kumpulkan dataset baru dengan tips:")
print("     - Berbagai angle (depan, samping, atas)")
print("     - Berbagai jarak (dekat, sedang, jauh)")
print("     - Berbagai pencahayaan")
print("     - Dengan background (orang, objek lain)")
print("     - Minimal 100-150 images per variant")
print("  3. Press [G] untuk generate train/val split")
print("  4. Train model: python train_variant_model.py")
print("=" * 60)