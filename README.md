# Hand Tracking Filter & Portal

Aplikasi filter kamera interaktif menggunakan gestur tangan (hand tracking) dengan MediaPipe dan OpenCV. Buat portal dengan jari untuk menerapkan filter pada area kamera.

Pratinjau memakai satu jendela berukuran tetap sesuai rasio kamera. Ukurannya 1280 × 720 untuk kamera 16:9 atau 960 × 720 untuk kamera 4:3. Seluruh bingkai terlihat. Deteksi tangan dan foto yang disimpan dengan **S** memakai resolusi kamera.

## Persyaratan Sistem
- Python 3.11 disarankan. Ketersediaan paket MediaPipe bergantung pada versi Python dan sistem operasi.
- Webcam

## Cara Install dan Menjalankan


### 1. Buat Virtual Environment (Opsional tapi Sangat Disarankan)
Gunakan virtual environment agar dependencies (library) tidak bentrok dengan project Python lainnya di komputer Anda.
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install semua library Python yang dibutuhkan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```

Atau jika Anda ingin menginstallnya secara manual satu per satu:
```bash
pip install opencv-python mediapipe numpy
```

### 3. Pastikan Model MediaPipe Tersedia
Aplikasi ini membutuhkan dua file model dari MediaPipe yang seharusnya sudah ada di dalam repository ini:
1. `hand_landmarker.task` (Untuk mendeteksi titik pada tangan)
2. `selfie_segmenter.tflite` (Untuk filter Galaxy / memisahkan background)

Jika file tersebut belum ada, pastikan untuk meletakkannya di dalam folder yang sama dengan file `main.py`.

### 4. Jalankan Aplikasi
Jalankan script utamanya dengan mengetik:
```bash
python main.py
```
*(Gunakan `python3 main.py` jika Anda menggunakan macOS/Linux dan tidak menggunakan virtual environment)*

## Cara Penggunaan Fitur

- **Membuka Portal:** Gunakan ujung telunjuk dan jempol dari **kedua** tangan Anda di depan kamera (total 4 jari). Sebuah portal berbentuk persegi empat akan terbentuk di antara keempat jari Anda, dan efek filter akan muncul di dalamnya.
- **Mode semua jari:** Tekan **`m`** untuk beralih ke portal yang menggunakan ujung kelima jari di kedua tangan. Batas portal mengikuti titik-titik terluar; tekan `m` lagi untuk kembali ke mode awal.
- **Mode dua bidang:** Tekan **`c`** atau kepalkan kedua tangan untuk beralih ke dua bidang yang memakai filter berurutan. Kepalan harus dilepas sebelum dapat mengganti mode lagi. Ini adalah efek dua bidang 2D, bukan rekonstruksi 3D.
- **Satu tangan:** Lima ujung jari satu tangan dapat membentuk portal ketika hanya satu tangan terdeteksi.
- **Mengganti Filter:** Ada beberapa cara untuk mengganti filter yang sedang aktif:
  - Sentuhkan/dekatkan ujung jempol dan jari kelingking Anda.
  - Atau, dekatkan ujung telunjuk dari kedua tangan Anda.
  - Tekan **`n`** untuk filter berikutnya atau **`p`** untuk sebelumnya.
- **Simpan foto:** Tekan **`s`** untuk menyimpan frame saat ini sebagai PNG di direktori tempat aplikasi dijalankan.
- **Menutup Aplikasi:** Tekan tombol **`q`** saat jendela kamera aktif, atau tutup jendela dengan tombol **X**.

Jika kamera tidak terbuka, periksa izin kamera dan tutup aplikasi lain yang sedang memakainya. Jika virtual environment tidak lagi berfungsi setelah Python dipindahkan atau diperbarui, buat ulang virtual environment dan instal kembali dependencies.

Filter yang tersedia mencakup Mono, Dual-Tone, Pixelate, Invert, Sepia, Blur, Thermal, Sketch, Glitch, Neon, Galaxy, Red, Cartoon, Rainbow, Water, Kaleidoscope, dan Dimension. Galaxy memerlukan model segmentasi; filter tersebut tidak ditampilkan jika model gagal dimuat. Efek Water, Rainbow, dan Dimension bergerak seiring waktu.

---
**Catatan:** Pastikan ruangan memiliki pencahayaan yang cukup agar deteksi tangan dari kamera dapat bekerja dengan optimal.
  
