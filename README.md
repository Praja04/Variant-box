# 📦 Pallet Variant & Stack Detection System

Sistem Computer Vision berbasis **YOLOv8** untuk deteksi varian kardus, validasi susunan palet, dan monitoring tinggi tumpukan (*stack height*) secara real-time melalui kamera webcam maupun CCTV RTSP industri.

---

## 🎯 Fitur Utama

- **Real-Time Variant Classification**: Klasifikasi varian kardus secara cepat dan akurat menggunakan model YOLOv8 kustom.
- **Pallet Stacking & QC Validation**: Memvalidasi susunan kardus per layer dan urutan tumpukan berdasarkan aturan produksi.
- **Dukungan Multi-Input**:
  - USB Webcam (inspeksi meja / manual QC)
  - IP CCTV / RTSP Stream (monitoring jalur palletizing otomatis)
- **Dataset Collection & Training Pipeline**: Dilengkapi tool pengambilan dataset otomatis (auto-capture, perspective correction) dan notebook training Google Colab (GPU T4).

---

## 🏷️ Varian Kardus yang Didukung

| Varian | Deskripsi | Standar Per Layer | Total Layer | Kapasitas Palet |
| :--- | :--- | :---: | :---: | :---: |
| **Sachet_20gr_LunchBox** | Sachet 20gr Lunch Box | 14 box | 6 layer | 84 box |
| **Variant_YB77** | Variant YB 77 | 20 box | 6 layer | 120 box |
| **Variant_BB725** | Variant BB 725 | 13 box | 7 layer | 91 box |

---

## 📁 Struktur Direktori

```text
project/
├── codes/
│   ├── pallet/                       # Modul utama Pallet Variant
│   │   ├── pallet_variant_detector_v2.py   # Deteksi varian & validasi layer (v2)
│   │   ├── pallet_variant_detector.py      # Deteksi varian (v1)
│   │   ├── pallet_dataset_collector_v2.py  # Kolektor dataset v2
│   │   ├── pallet_dataset_collector.py     # Kolektor dataset v1
│   │   ├── train_variant_model_v2.py       # Script training lokal v2
│   │   ├── train_variant_model.py          # Script training lokal v1
│   │   ├── auto_camera_detect.py           # Utilitas auto-detect ID webcam
│   │   └── reset_dataset_and_model.py      # Reset data eksperimen
│   ├── python/                       # Modul deteksi real-time & CCTV
│   │   ├── detect_variant_webcam.py        # Deteksi varian kardus via Webcam
│   │   ├── pallet_stack_detector.py        # Deteksi tumpukan palet via RTSP
│   │   ├── detection_pallet.py             # Pallet QC & validation
│   │   └── dataset_collector_cctv.py       # Kolektor dataset dari CCTV RTSP
│   ├── generate_notebook.py          # Generator notebook Google Colab
│   ├── prepare_colab.py              # Script packaging dataset untuk Colab
│   ├── Train_Pallet_YOLO_Colab.ipynb # Notebook training Colab siap pakai
│   └── requirements.txt              # Daftar dependensi Python
├── laporan_evaluasi_model.md         # Laporan evaluasi performa model YOLO
├── legacy_non_pallet/                # Arsip script non-pallet (di-ignore oleh git)
└── .gitignore                        # Konfigurasi filter file repository
```

---

## 🚀 Instalasi & Persiapan

### 1. Clone Repositori
```bash
git clone https://github.com/Praja04/<nama-repository>.git
cd <nama-repository>
```

### 2. Buat Lingkungan Virtual & Install Dependensi
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r codes/requirements.txt
```

---

## 💻 Cara Penggunaan

### 1. Cek Index Kamera Webcam
Jika menggunakan USB webcam eksternal:
```bash
python codes/pallet/auto_camera_detect.py
```

### 2. Jalankan Deteksi Varian via Webcam
```bash
python codes/python/detect_variant_webcam.py
```
> **Shortcut Keyboard:**
> - `Q` : Keluar
> - `S` : Simpan screenshot hasil deteksi

### 3. Jalankan Detektor Varian & Validasi Susunan Palet
```bash
cd codes/pallet
python pallet_variant_detector_v2.py
```

### 4. Monitoring Tumpukan Palet via CCTV (RTSP)
Sesuaikan URL RTSP pada file `codes/python/pallet_stack_detector.py`, lalu jalankan:
```bash
python codes/python/pallet_stack_detector.py
```

### 5. Training Model Ulang di Google Colab
1. Siapkan dataset:
   ```bash
   python codes/prepare_colab.py
   ```
2. Upload file `dataset_for_colab.zip` ke Google Drive / Google Colab.
3. Buka dan jalankan cell pada `codes/Train_Pallet_YOLO_Colab.ipynb`.

---

## 📄 Lisensi & Pembuat
Dikembangkan untuk sistem QC Palletizing Industri.

