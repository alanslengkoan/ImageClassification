# Pedoman Singkat: Validasi SegFormer-B0 (Ground-Truth Mask & mAP50-95)

Tujuan: mengukur **seberapa tepat SegFormer-B0 menandai area jalan** sebelum foto diklasifikasi, dengan membandingkannya terhadap anotasi manual (ground-truth).

---

## Persiapan

| Kebutuhan | Keterangan |
|-----------|------------|
| Sistem | Windows 10/11 dengan **Ubuntu (WSL)** |
| Folder proyek | `D:\ImageClassification-CNN` |
| Internet | Untuk install library & unduh model SegFormer (±15 MB, sekali saja) |

Semua perintah dijalankan di **terminal Ubuntu**. Di Ubuntu, drive `D:` dibaca sebagai `/mnt/d`, jadi:

```
D:\ImageClassification-CNN   =   /mnt/d/ImageClassification-CNN
```

---

## File yang Dipakai

| File | Fungsi |
|------|--------|
| `train_models/validate_segformer.py` | Script utama — **ini yang dijalankan** |
| `train_models/segmentation_metrics.py` | Rumus IoU, Dice, mAP50-95 (dipanggil otomatis) |
| `train_models/test_segmentation_metrics.py` | Cek rumus sudah benar |

---

## Langkah-Langkah

### 1. Install library (sekali saja)

```bash
sudo apt update
sudo apt install -y python3-pip libgl1 libglib2.0-0
pip install --user --break-system-packages numpy opencv-python matplotlib pillow transformers labelme
pip install --user --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
```

Cek instalasi:

```bash
python3 -c "import torch, transformers, cv2; print('OK')"
```

Harus muncul: `OK`

### 2. Atur lokasi folder proyek (sekali saja)

```bash
cd /mnt/d/ImageClassification-CNN
sed -i -E "s#^(BASE_DIR *= *')[^']*ImageClassification(-CNN)?#\1/mnt/d/ImageClassification-CNN#" train_models/*.py
grep -n "BASE_DIR *=" train_models/validate_segformer.py
```

Harus muncul: `BASE_DIR = '/mnt/d/ImageClassification-CNN/train_models'`

> Jika folder proyek dipindah ke lokasi lain, ganti `/mnt/d/ImageClassification-CNN` pada perintah di atas dengan lokasi yang baru.

### 3. Cek rumus

```bash
cd /mnt/d/ImageClassification-CNN
python3 train_models/test_segmentation_metrics.py
```

Harus muncul: `✅ ALL 30 TESTS PASSED`

### 4. Ambil sampel gambar

```bash
cd /mnt/d/ImageClassification-CNN/train_models
python3 validate_segformer.py --prepare 20
```

Hasil: 20 foto per kelas (total 60) tersalin ke `D:\ImageClassification-CNN\train_models\segmentation_gt\images\` (folder `baik`, `sedang`, `berat`).

### 5. Anotasi area jalan (LabelMe)

```bash
cd /mnt/d/ImageClassification-CNN/train_models
labelme segmentation_gt/images/baik --labels jalan
```

1. Tekan `Ctrl+N` → klik mengelilingi tepi jalan → klik titik awal untuk menutup polygon
2. Pilih label **`jalan`**
3. Tekan `D` untuk gambar berikutnya (`A` untuk kembali). Anotasi tersimpan otomatis sebagai `.json` di samping gambar.
4. Setelah selesai, tutup LabelMe lalu ulangi untuk folder `sedang` dan `berat`:

```bash
labelme segmentation_gt/images/sedang --labels jalan
labelme segmentation_gt/images/berat  --labels jalan
```

> Jika muncul `unrecognized arguments`, pastikan perintahnya persis seperti di atas. Opsi lama `--nodata` dan `--autosave` sudah tidak ada di LabelMe versi baru (v7) karena keduanya sekarang sudah aktif secara bawaan.

#### Cara cepat: anotasi dibantu AI (SAM 2)

Daripada mengklik tepi jalan satu per satu, LabelMe bisa membuat polygon otomatis memakai model **SAM 2**. Fitur ini berjalan di CPU, jadi tidak perlu GPU.

**a. Unduh model (sekali saja, butuh internet)**

1. Di toolbar atas ada panel **AI-Assisted Annotation**. Panel ini baru aktif setelah memilih mode **AI-Points** atau **AI-Box** di toolbar kiri, paling bawah.
2. Buka dropdown model → pilih **Download a model…** → unduh **Sam2 (balanced)** (±330 MB). Untuk laptop yang lambat, pilih **Sam2 (speed)** (±185 MB).
3. Setelah unduhan selesai, pilih model tersebut di dropdown.

> Jika tampilan LabelMe berbahasa Indonesia, menunya bernama **Anotasi berbasis AI**, **Unduh model…**, dan **Kelola model…**.
> Di LabelMe v6, model langsung terunduh otomatis saat titik pertama diklik.
> Model disimpan di `~/.cache/osam` (Ubuntu) atau `C:\Users\<nama>\.cache\osam` (Windows), sehingga cukup diunduh sekali.

**b. Pastikan format output = `Polygon`**

Di panel yang sama ada dropdown kedua (format output). Pastikan isinya **`Polygon`**, karena itu nilai bawaannya.

> ⚠️ **Jangan pilih `Mask`.** Script validasi hanya membaca polygon. Anotasi berformat `Mask` akan diabaikan dan gambar dianggap tidak punya jalan.

**c. Anotasi**

| Mode | Cara Pakai |
|------|------------|
| **AI-Points** (disarankan) | Klik di **atas permukaan jalan** (1–3 titik). `Shift`+klik di area yang **bukan** jalan bila ikut terpilih. Tekan `Enter` untuk selesai. |
| **AI-Box** | Tarik kotak yang mengelilingi seluruh jalan, lalu lepas mouse. Polygon langsung jadi. |

Setelah polygon jadi:

1. Pilih label **`jalan`**.
2. **Periksa hasilnya.** Jika ada bagian yang meleset, tekan `Ctrl+J` (Edit) lalu geser titik polygon, atau tekan `Ctrl+Z` dan ulangi.
3. Tekan `D` untuk gambar berikutnya.

Catatan:

- Gambar pertama terasa lambat (beberapa detik) karena AI sedang menganalisis gambar. Klik berikutnya akan jauh lebih cepat.
- Jika jalan terpecah menjadi beberapa polygon, tidak masalah. Asalkan semuanya diberi label `jalan`, semuanya otomatis digabung.
- AI hanya membantu menggambar. **Setiap hasil tetap wajib dicek dan dirapikan manual** sesuai aturan di bawah, karena anotasi ini menjadi kunci jawaban.

**Aturan anotasi:**

| Termasuk jalan ✅ | Bukan jalan ❌ |
|-------------------|----------------|
| Aspal / beton | Trotoar, tanah, rumput |
| Lubang, retak, tambalan, genangan | Kendaraan, orang |
| Marka jalan | Bangunan, pohon, langit |

> **Jendela LabelMe tidak muncul?** Tampilan aplikasi dari Ubuntu (WSL) hanya didukung di Windows 11. Untuk Windows 10, install [Python untuk Windows](https://www.python.org/downloads/windows/) (centang *Add python.exe to PATH*), lalu jalankan di **PowerShell**:
>
> ```powershell
> pip install labelme
> labelme D:\ImageClassification-CNN\train_models\segmentation_gt\images\baik --labels jalan
> ```
>
> Hasil `.json`-nya sama dan bisa langsung dipakai di Langkah 6.

### 6. Jalankan validasi

```bash
cd /mnt/d/ImageClassification-CNN/train_models
python3 validate_segformer.py
```

Hasil tersimpan di `D:\ImageClassification-CNN\train_models\dataset\output\segmentation_validation\`:

| File | Isi |
|------|-----|
| `segformer_validation_report.txt` | Tabel hasil utama |
| `segformer_ap_iou.png` | Grafik AP & IoU |
| `segformer_per_class_metrics.png` | Grafik per kelas |
| `preview\` | Contoh overlay: 🟩 benar, 🟥 salah tandai, 🟦 terlewat |

Folder ini bisa dibuka langsung dari **File Explorer** Windows.

---

## Cara Membaca Hasil

| Metrik | Arti Sederhana |
|--------|----------------|
| **IoU** | Seberapa pas area jalan prediksi dengan aslinya (0–1, makin tinggi makin baik) |
| **Dice** | Mirip IoU, nilainya sedikit lebih tinggi |
| **Precision** | Yang ditandai jalan, berapa yang benar jalan? |
| **Recall** | Jalan yang asli, berapa yang berhasil ditandai? |
| **AP50** | Persentase gambar dengan IoU ≥ 0.50 (syarat longgar) |
| **mAP50-95** | Rata-rata AP dari syarat longgar (0.50) sampai sangat ketat (0.95) — **metrik utama** |

- **RAW** = hasil mentah SegFormer, **POST** = hasil setelah dirapikan `clean_objects.py` (yang benar-benar dipakai).
- Jika POST > RAW, berarti proses perapian mask terbukti membantu.
- **Recall** paling penting: jika rendah, ada kerusakan jalan yang ikut terhapus sebelum klasifikasi.
- mAP50-95 lebih rendah dari AP50 itu **normal**.

---

## Masalah Umum

| Pesan | Solusi |
|-------|--------|
| `pip: command not found` | `sudo apt install -y python3-pip` |
| `externally-managed-environment` | Tambahkan `--user --break-system-packages` pada `pip install` |
| `no such option: --break-system-packages` | Ubuntu versi lama — hapus `--break-system-packages` dari perintah |
| `libGL.so.1: cannot open shared object file` | `sudo apt install -y libgl1 libglib2.0-0` |
| `No module named 'numpy'` / `'torch'` | Library belum terpasang, ulangi Langkah 1 |
| `labelme: command not found` | `export PATH="$HOME/.local/bin:$PATH"` lalu coba lagi |
| `labelme: error: unrecognized arguments: --nodata --autosave` | Hapus kedua opsi itu: `labelme segmentation_gt/images/baik --labels jalan` |
| Panel **AI-Assisted Annotation** abu-abu | Pilih dulu mode **AI-Points** / **AI-Box** di toolbar kiri |
| Saat klik AI-Points/AI-Box muncul jendela **Settings** | Model belum diunduh → unduh **Sam2 (balanced)** di jendela itu, lalu pilih modelnya |
| Unduhan model gagal | Cek internet, lalu ulangi dari menu yang sama |
| AI sangat lambat | Pakai model **Sam2 (speed)** |
| Sudah dianotasi pakai AI tapi dianggap kosong | Format output AI masih `Mask` → ganti `Polygon`, lalu anotasi ulang gambar tersebut |
| `No such file or directory: /mnt/d/...` | Pastikan folder proyek ada di `D:\ImageClassification-CNN` |
| `Folder tidak ditemukan` saat `--prepare` | Langkah 2 belum dijalankan |
| `No module named 'segmentation_metrics'` | Jalankan dari dalam folder `train_models` |
| `Tidak ada mask ground-truth` | Anotasi belum ada / nama `.json` tidak sama dengan nama gambar |
| `N gambar belum punya mask` | Ada gambar yang belum dianotasi |

Penjelasan lengkap rumus metrik ada di `README.md` bagian 11.

