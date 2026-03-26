# 🔍 Face Detector Script

Script Python untuk mendeteksi wajah dalam gambar dan mengurutkannya ke dalam folder berdasarkan jumlah wajah yang terdeteksi.

## 📁 Struktur Output

Script akan membuat dan memindahkan gambar ke folder berikut:

```
output_folder/
├── no_face/      # Gambar tanpa wajah terdeteksi (0 wajah)
├── 1_face/       # Gambar dengan tepat 1 wajah
└── multi_face/   # Gambar dengan 2 atau lebih wajah
```

## 🚀 Instalasi

### 1. Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
cd face_detector
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
.\venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📖 Cara Penggunaan

### Penggunaan Dasar

```bash
# Proses gambar dan pindahkan ke subfolder di folder yang sama
python face_detector.py /path/to/images

# Proses gambar dan pindahkan ke folder output terpisah
python face_detector.py /path/to/images --output /path/to/output

# Copy file alih-alih memindahkan
python face_detector.py /path/to/images --copy

# Mode verbose untuk melihat detail posisi wajah
python face_detector.py /path/to/images --verbose
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Folder output untuk gambar yang sudah diurutkan |
| `--copy` | `-c` | Copy file alih-alih memindahkannya |
| `--verbose` | `-v` | Tampilkan info detail deteksi wajah |

## 📷 Format Gambar yang Didukung

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)
- WebP (.webp)
- TIFF (.tiff, .tif)

## 📊 Contoh Output

```
==================================================
🔍 FACE DETECTOR
==================================================
  📂 Source:  /Users/user/photos
  📁 Output:  /Users/user/photos
  📋 Mode:    Move
==================================================

🔄 Loading face detector...
✅ Face detector loaded

📸 Found 5 image(s) to process

[1/5] Processing: photo1.jpg
  🙂 Detected 1 face(s) → 1_face/
[2/5] Processing: photo2.jpg
  👥 Detected 3 face(s) → multi_face/
[3/5] Processing: landscape.jpg
  😶 Detected 0 face(s) → no_face/
[4/5] Processing: selfie.png
  🙂 Detected 1 face(s) → 1_face/
[5/5] Processing: group.jpg
  👥 Detected 5 face(s) → multi_face/

==================================================
📊 SUMMARY
==================================================
  📁 Total images processed: 5
  😶 No face detected:       1
  🙂 One face detected:      2
  👥 Multiple faces:         2
  ⚠️  Errors/Skipped:         0
==================================================

✅ Done!
```

## 🔧 Troubleshooting

### Error: "Could not read image"
- Pastikan file gambar tidak corrupt
- Cek permission file

### Error: "Haar Cascade file not found"
- Pastikan OpenCV terinstall dengan benar
- Reinstall: `pip install --force-reinstall opencv-python`

### Deteksi Wajah Tidak Akurat
- Script menggunakan Haar Cascade yang cepat tapi mungkin kurang akurat untuk:
  - Wajah yang miring/rotasi
  - Wajah dengan pencahayaan buruk
  - Wajah yang terhalang sebagian
- Untuk akurasi lebih tinggi, pertimbangkan menggunakan library `face_recognition` atau model deep learning

## 📝 License

MIT License
