# Penjelasan: Mengapa Hasil Training Tidak Sama?

## 1. Kenapa Laptop A dan Laptop B Beda Hasil?

Walaupun dataset dan kode yang digunakan **sama persis**, kedua laptop memiliki **CPU yang berbeda** (Intel Gen 11 vs Gen 12). Keduanya menghitung matematika yang sama, tetapi hasilnya **sedikit berbeda di belakang koma**.

Contoh sederhana:

```
Laptop A menghitung: 1.0 / 3.0 = 0.33333334
Laptop B menghitung: 1.0 / 3.0 = 0.33333335
```

Perbedaan kecil ini **tidak terasa dalam 1 hitungan**. Namun dalam proses training, terjadi **jutaan hitungan per epoch × puluhan epoch**. Perbedaan kecil itu terakumulasi seperti **bola salju** — semakin lama semakin besar.

**Analogi:** Dua orang berjalan dari titik yang sama, tetapi satu belok 0.1° ke kanan. Setelah berjalan 10 km, keduanya sudah berada di **tempat yang sangat berbeda**.

---

## 2. Kenapa Laptop A Selalu ~77% dan Susah Naik?

### a. Dataset Terlalu Kecil

Dataset hanya berisi **840 gambar training** (280 per kelas), sedangkan model MobileNetV2 memiliki **2.6 juta parameter**. Ini seperti belajar untuk ujian hanya dengan membaca ringkasan — bisa hafal ringkasannya, tetapi belum tentu paham konsepnya.

### b. Dua Kelas yang Sangat Mirip

Kelas **"Sedang"** dan **"Berat"** secara visual sangat mirip. Seperti diminta membedakan hujan "sedang" dan hujan "deras" dari foto — batasnya sangat abu-abu. Ini terbukti dari data:

| Kelas | Precision | Recall | Keterangan |
|-------|-----------|--------|------------|
| Baik | 0.83 | 0.95 | Mudah dikenali |
| Berat | 0.71 | 0.97 | Model terlalu sering memprediksi kelas ini |
| Sedang | 0.84 | 0.40 | Sering salah dikira "Berat" |

### c. Overfitting (Menghafal, Bukan Memahami)

Model mencapai **98% akurasi di data training** (hafal semua soal latihan) tetapi hanya **77% di test set** (soal baru). Ini disebut **overfitting** — seperti siswa yang hafal jawaban buku tetapi gagal saat soal diubah sedikit.

**Bukti overfitting dari 3 kali percobaan:**

| Percobaan | Train Accuracy | Val Accuracy | Test Accuracy (TTA) | Gap Train-Val |
|-----------|---------------|-------------|---------------------|---------------|
| v1 | ~90% | 79.58% | 77.50% | ~10% |
| v2 | ~88% | 82.50% | 76.67% | ~6% |
| v3 | ~98% | 79.58% | 77.50% | ~19% |

Gap train-val yang besar menunjukkan model menghafal data training, bukan belajar pola umum.

---

## 3. Kenapa Ubah Hyperparameter Tidak Banyak Membantu?

Hyperparameter itu seperti **teknik belajar** (belajar pagi/malam, pakai stabilo/tidak, baca cepat/pelan).

Masalah utamanya bukan teknik belajarnya, tetapi **bukunya kurang tebal**. Tidak peduli teknik apapun yang digunakan, jika materinya hanya 840 halaman dan 2 topik sangat mirip, hasilnya akan tetap mentok di kisaran yang sama.

Itu sebabnya dari v1 sampai v3, hasil test accuracy selalu berada di kisaran **76-78%** — karena itu merupakan **batas kemampuan** model ini dengan jumlah data yang tersedia.

### Perubahan yang Sudah Dicoba

| Versi | Perubahan Utama | Hasil TTA |
|-------|----------------|-----------|
| v1 | Tambah reproducibility (deterministic mode) | 77.50% |
| v2 | Tuning: fine-tune 50 layers, dropout 0.4/0.3, class weight | 76.67% |
| v3 | Relax deterministic, LR 1e-4, fine-tune 80 layers | 77.50% |

Hasil selalu di kisaran yang sama karena **bottleneck-nya adalah data, bukan hyperparameter**.

---

## 4. Solusi yang Tersedia

| Solusi | Analogi | Dampak | Keterangan |
|--------|---------|--------|------------|
| **Tambah data** | Buku lebih tebal | ⭐⭐⭐ Paling efektif | Terutama kelas "Sedang" yang paling sulit |
| **Model lebih sederhana** | Otak yang tidak terlalu menghafal | ⭐⭐ Sedang dicoba (v4) | Kurangi parameter agar tidak overfit |
| **Augmentasi lebih kuat** | Baca buku dari sudut berbeda | ⭐ Membantu sedikit | Menambah variasi data secara artifisial |

### Pendekatan v4 (Anti-Overfitting Agresif)

| Parameter | Sebelum (v3) | Sesudah (v4) | Alasan |
|-----------|-------------|-------------|--------|
| Fine-tune Layers | 80 | **20** | Drastis kurangi parameter yang berubah |
| Head Layers | Dense(256) + Dense(128) | **Dense(128) saja** | Hapus 1 layer, kurangi parameter |
| Dropout | 0.3 / 0.2 | **0.5** | Lebih agresif cegah overfitting |
| L2 Regularization | 0.0003 | **0.001** | 3x lebih kuat |
| Phase 2 LR | 1e-4 | **3e-5** | Lebih rendah untuk stabilitas |

**Target v4:** Gap train-val turun dari 19% ke <5%, sehingga test accuracy lebih mendekati val accuracy.

---

## 5. Kesimpulan

> Perbedaan hasil antar laptop disebabkan oleh **perbedaan floating-point rounding di CPU yang berbeda**, yang terakumulasi selama jutaan operasi dalam proses training.
>
> Hasil yang mentok di ~77% pada Laptop A disebabkan oleh **keterbatasan jumlah data** (840 gambar) untuk model dengan jutaan parameter, ditambah **kelas "Sedang" dan "Berat" yang sangat mirip** secara visual.
>
> **Tuning hyperparameter saja tidak cukup** — solusi paling efektif adalah menambah jumlah dan kualitas data training, terutama untuk kelas yang sulit dibedakan.
