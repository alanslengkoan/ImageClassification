# Road Damage Detector (Aplikasi Web Django)

Aplikasi web untuk **klasifikasi kerusakan jalan** berbasis citra menggunakan model
CNN (ResNet50) yang sudah dilatih. Pengguna mengunggah satu atau beberapa gambar
jalan, lalu sistem mengklasifikasikannya ke dalam 3 kelas:

| Kelas | Indikator | Keterangan |
|-------|-----------|-----------|
| **Baik**   | 🟢 | Kondisi jalan baik, tidak ada kerusakan signifikan |
| **Sedang** | 🟡 | Kerusakan ringan hingga sedang pada permukaan jalan |
| **Berat**  | 🔴 | Kerusakan berat, perlu perbaikan segera |

---

## 1. Prasyarat

- **Python 3.12** (atau 3.10+)
- **pip**
- Disarankan menggunakan **virtual environment**
- Ruang disk ± 1 GB (TensorFlow + model ± 173 MB)

---

## 2. Struktur Folder

Struktur penting yang berhubungan dengan aplikasi web:

```
ImageClassification/
├── models/                          # << FOLDER MODEL (lihat Bagian 3)
│   └── resnet50_3class_best.h5      # << FILE MODEL yang dibaca aplikasi
│
└── road_detector/                   # Root proyek Django
    ├── manage.py
    ├── requirements.txt
    ├── media/                        # Hasil upload gambar tersimpan di sini
    ├── detector/                     # App utama (views, urls, template)
    │   ├── views.py                  # Logika load model & prediksi
    │   ├── urls.py
    │   └── templates/detector/
    │       └── index.html
    └── road_detector/                # Konfigurasi proyek
        ├── settings.py               # Berisi MODEL_PATH
        ├── urls.py
        └── wsgi.py
```

---

## 3. Folder & Penamaan Model (PENTING)

Aplikasi membaca model dari path yang didefinisikan di
`road_detector/road_detector/settings.py`:

```python
MODEL_PATH = BASE_DIR.parent / 'models' / 'resnet50_3class_best.h5'
```

Karena `BASE_DIR` menunjuk ke folder `road_detector/` (lokasi `manage.py`),
maka `BASE_DIR.parent` adalah folder `ImageClassification/`.

**Letakkan file model dengan ketentuan berikut:**

- **Folder:** `ImageClassification/models/`
  (folder `models/` berada **satu tingkat di atas** folder `road_detector/`)
- **Nama file (WAJIB sama persis):** `resnet50_3class_best.h5`
- **Path lengkap akhir:**
  `ImageClassification/models/resnet50_3class_best.h5`

> Jika nama file atau lokasi berbeda, aplikasi akan menampilkan error
> `Model tidak ditemukan`. Pastikan penamaan **persis** atau ubah `MODEL_PATH`
> di `settings.py` sesuai lokasi model Anda.

### Cara memperoleh file model

1. **Sudah tersedia** — file model biasanya sudah ada di `ImageClassification/models/resnet50_3class_best.h5`.
2. **Melatih ulang** — jalankan skrip training, lalu salin hasilnya:
   ```bash
   # dari folder ImageClassification/
   python3 train_models/resnet50.py
   cp train_models/dataset/output/resnet50_3class_best.h5 models/
   ```

---

## 4. Instalasi

Jalankan dari folder `road_detector/`:

```bash
# 1. Masuk ke folder proyek Django
cd road_detector

# 2. (Disarankan) buat & aktifkan virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Pasang dependencies
pip install -r requirements.txt
```

---

## 5. Menjalankan Aplikasi

Pastikan file model sudah berada di `ImageClassification/models/resnet50_3class_best.h5`
(lihat Bagian 3), lalu jalankan server pengembangan dari folder `road_detector/`:

```bash
python3 manage.py runserver
```

Buka di browser:

```
http://127.0.0.1:8000/
```

Untuk dapat diakses dari perangkat lain di jaringan yang sama:

```bash
python3 manage.py runserver 0.0.0.0:8000
```

Hentikan server dengan `CTRL + C`.

---

## 6. Cara Penggunaan

1. Buka `http://127.0.0.1:8000/`.
2. Pilih **1 sampai 10 gambar** jalan (format umum: JPG/PNG).
3. Klik tombol unggah/analisis.
4. Sistem menampilkan untuk tiap gambar:
   - Kelas prediksi (Baik / Sedang / Berat)
   - Tingkat keyakinan (confidence)
   - Distribusi probabilitas semua kelas

> **Catatan:** Model di-load secara *lazy* (saat upload pertama), sehingga
> prediksi pertama akan sedikit lebih lambat. Upload berikutnya lebih cepat.

### Batasan upload (di `settings.py`)

| Pengaturan | Nilai |
|-----------|-------|
| Maks total upload | 100 MB |
| Maks ukuran per file | 10 MB |
| Maks jumlah file sekaligus | 10 file |

---

## 7. Troubleshooting

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `Model tidak ditemukan: .../models/resnet50_3class_best.h5` | File model tidak ada / salah nama / salah folder | Pastikan file `resnet50_3class_best.h5` ada di `ImageClassification/models/` (Bagian 3) |
| `Django tidak terinstall` | Dependencies belum dipasang | Jalankan `pip install -r requirements.txt` |
| Error saat load TensorFlow | Versi TF tidak cocok | Gunakan `tensorflow==2.21.0` sesuai `requirements.txt` |
| Port 8000 sudah dipakai | Ada server lain berjalan | Jalankan di port lain, mis. `python3 manage.py runserver 8001` |
| Prediksi pertama lambat | Model baru di-load (lazy load) | Normal, tunggu beberapa detik |

---

## 8. Catatan Teknis

- **Preprocessing:** gambar di-resize ke `224x224` lalu memakai
  `resnet50.preprocess_input` (Caffe-style) — sesuai cara model dilatih.
- **Urutan kelas:** `['baik', 'berat', 'sedang']` (alfabetis, mengikuti
  `ImageDataGenerator` saat training). Jangan diubah agar prediksi tetap benar.
- **Penyimpanan upload:** gambar yang diunggah disimpan di folder `media/uploads/`.
