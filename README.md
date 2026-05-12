# ImageClassification# Penjelasan Perbandingan Model: ResNet50 vs MobileNetV2

## Klasifikasi Kerusakan Jalan Berbasis Citra (3 Kelas)

---

## 1. Ringkasan Hasil Perbandingan

| Metrik | ResNet50 | MobileNetV2 |
|--------|----------|-------------|
| **Test Accuracy (TTA)** | 79.17% | **85.00%** |
| **Test Accuracy (Standard)** | 78.33% | 83.33% |
| **Test Loss** | 0.8366 | ≈ 0.55 |
| **Best Val Accuracy** | 80.83% | 85.42% |
| **Precision (Baik)** | 0.86 | 0.88 |
| **Recall (Baik)** | 0.95 | 0.92 |
| **F1-Score (Baik)** | 0.90 | 0.90 |
| **Precision (Berat)** | 0.78 | 0.80 |
| **Recall (Berat)** | 0.80 | 0.85 |
| **F1-Score (Berat)** | 0.79 | 0.82 |
| **Precision (Sedang)** | 0.71 | 0.82 |
| **Recall (Sedang)** | 0.62 | 0.78 |
| **F1-Score (Sedang)** | 0.67 | 0.80 |

### Kesimpulan Utama
MobileNetV2 **unggul +5.83%** dibanding ResNet50 dalam test accuracy. Keunggulan paling signifikan terlihat pada kelas **"Sedang"** dimana MobileNetV2 mencapai recall 78% sementara ResNet50 hanya 62%.

---

## 2. Arsitektur Model

### 2.1 MobileNetV2

| Aspek | Detail |
|-------|--------|
| **Backbone** | MobileNetV2 (pretrained ImageNet) |
| **Total Parameter Backbone** | ~3.4 juta |
| **Total Layer** | 155 layer |
| **Arsitektur Inti** | Depthwise Separable Convolution + Inverted Residuals |
| **Head Classifier** | GAP → Dense(256) → BN → Dropout(0.3) → Dense(128) → BN → Dropout(0.2) → Softmax |
| **Fine-tune Layers** | 80 layer terakhir (51.6% dari total) |

**Karakteristik:**
- Dirancang untuk perangkat mobile dengan komputasi terbatas
- Menggunakan **Depthwise Separable Convolution** yang memisahkan operasi spasial dan channel
- **Inverted Residual Block**: expand → depthwise → project (bottleneck)
- Efisien dalam jumlah parameter namun tetap mampu mengekstrak fitur yang kaya
- Lebih cocok untuk dataset kecil karena risiko overfitting lebih rendah

### 2.2 ResNet50

| Aspek | Detail |
|-------|--------|
| **Backbone** | ResNet50 (pretrained ImageNet) |
| **Total Parameter Backbone** | ~25.6 juta |
| **Total Layer** | 175 layer |
| **Arsitektur Inti** | Residual Blocks dengan Skip Connections |
| **Head Classifier** | GAP → BN → Dense(128) → Dropout(0.5) → Softmax |
| **Fine-tune Layers** | 30 layer terakhir (17.1% dari total) |

**Karakteristik:**
- Arsitektur deep learning klasik yang sangat powerful
- **Skip Connections** memungkinkan gradient mengalir langsung, mengatasi vanishing gradient
- Jumlah parameter **7.5x lebih besar** dari MobileNetV2
- Membutuhkan **lebih banyak data** untuk menghindari overfitting
- Lebih cocok untuk dataset besar (puluhan ribu hingga jutaan gambar)

---

## 3. Distribusi Data

### 3.1 Komposisi Dataset

| Split | Baik | Berat | Sedang | Total |
|-------|------|-------|--------|-------|
| **Train (70%)** | 280 | 280 | 280 | **840** |
| **Validation (20%)** | 80 | 80 | 80 | **240** |
| **Test (10%)** | 40 | 40 | 40 | **120** |
| **TOTAL** | 400 | 400 | 400 | **1.200** |

### 3.2 Analisis Jumlah Data

#### Mengapa 1.200 Gambar Tergolong Dataset Kecil?

Dalam konteks deep learning dengan arsitektur pretrained:
- **Dataset kecil**: < 5.000 gambar
- **Dataset sedang**: 5.000 – 50.000 gambar
- **Dataset besar**: > 50.000 gambar

Dengan hanya **840 gambar training** (280 per kelas), kedua model menghadapi tantangan:

1. **Risiko Overfitting Tinggi** — Model mudah menghafal data training
2. **Variasi Terbatas** — 280 gambar per kelas tidak cukup merepresentasikan semua variasi kerusakan jalan di dunia nyata
3. **Generalisasi Sulit** — Model harus mampu mengklasifikasi gambar yang belum pernah dilihat dengan hanya sedikit contoh

#### Dampak Jumlah Data Terhadap Masing-Masing Model

| Faktor | ResNet50 (25.6M params) | MobileNetV2 (3.4M params) |
|--------|------------------------|--------------------------|
| **Rasio Data:Parameter** | 840 : 25.600.000 = **1:30.476** | 840 : 3.400.000 = **1:4.048** |
| **Risiko Overfitting** | **SANGAT TINGGI** | Moderat |
| **Kebutuhan Data Ideal** | > 50.000 gambar | > 5.000 gambar |
| **Gap Train-Val** | ~15% (train 95%, val 80%) | ~8% (train 92%, val 85%) |

**Interpretasi:**
- ResNet50 memiliki rasio data:parameter yang sangat timpang (1 gambar untuk 30.476 parameter). Ini menyebabkan model sangat mudah menghafal.
- MobileNetV2 dengan 7.5x lebih sedikit parameter, lebih mampu memanfaatkan 840 gambar training secara efisien.

---

## 4. Strategi Augmentasi Data

### 4.1 Mengapa Augmentasi Diperlukan?

Dengan hanya 840 gambar training, augmentasi data berfungsi untuk:
1. **Memperbanyak variasi** — Seolah-olah model melihat gambar yang berbeda setiap epoch
2. **Mencegah overfitting** — Model tidak bisa menghafal gambar yang selalu berubah
3. **Meningkatkan generalisasi** — Model belajar fitur yang invariant terhadap transformasi

### 4.2 Perbandingan Augmentasi

| Parameter Augmentasi | MobileNetV2 | ResNet50 | Alasan Berbeda |
|---------------------|-------------|----------|----------------|
| **rotation_range** | 15° | 25° | ResNet50 butuh augmentasi lebih kuat untuk lawan overfitting |
| **width_shift_range** | 0.10 | 0.15 | Pergeseran lebih besar untuk variasi posisi |
| **height_shift_range** | 0.10 | 0.15 | Sama seperti width_shift |
| **shear_range** | 0.08 | 0.12 | Distorsi perspektif lebih besar |
| **zoom_range** | 0.10 | 0.20 | Variasi skala objek dalam frame |
| **brightness_range** | [0.90, 1.10] | [0.85, 1.15] | Variasi pencahayaan lebih luas |
| **horizontal_flip** | True | True | Sama — kerusakan jalan simetris |
| **vertical_flip** | False | False | Jalan tidak boleh dibalik vertikal |

### 4.3 Mengapa ResNet50 Butuh Augmentasi Lebih Kuat?

```
Kapasitas Model Besar → Mudah Menghafal → Butuh Augmentasi Lebih Agresif
     25.6M params          Train 95%+            rotation=25°, zoom=0.20
```

Dengan augmentasi yang sama ringannya seperti MobileNetV2 (rotation=15, zoom=0.10):
- ResNet50 mengalami **overfitting parah**: train accuracy 95%+ vs val accuracy 80%
- Gap 15% menunjukkan model menghafal data, bukan belajar pola umum

Augmentasi yang lebih kuat memaksa ResNet50 untuk:
- Tidak bergantung pada posisi pixel yang eksak
- Belajar fitur yang lebih robust terhadap variasi
- Mengurangi gap antara training dan validation

### 4.4 Efektivitas Augmentasi Terhadap Jumlah Data

| Tanpa Augmentasi | Dengan Augmentasi |
|-----------------|-------------------|
| 840 gambar statis per epoch | 840 gambar **berbeda** setiap epoch |
| Model melihat gambar yang sama berulang | Model melihat variasi baru setiap iterasi |
| Overfitting dalam < 10 epoch | Training lebih stabil hingga 30-60 epoch |
| Efektif hanya melatih dengan 840 unique samples | Efektif melatih dengan **ribuan** variasi virtual |

**Estimasi efek augmentasi:**
- Dengan rotation=25°, shift=0.15, zoom=0.20, flip=True
- Setiap gambar bisa menghasilkan ratusan variasi unik
- 840 gambar × augmentasi ≈ **puluhan ribu** variasi virtual per epoch

---

## 5. Faktor-Faktor yang Mempengaruhi Performa

### 5.1 Faktor Utama: Kapasitas Model vs Jumlah Data

| Prinsip | Penjelasan |
|---------|-----------|
| **Bias-Variance Tradeoff** | Model besar (ResNet50) memiliki low bias tapi high variance → overfitting pada data kecil |
| **Occam's Razor** | Model yang lebih sederhana (MobileNetV2) lebih baik ketika data terbatas |
| **Proportionalitas** | Jumlah parameter harus proporsional dengan jumlah data training |

### 5.2 Faktor Regularisasi

| Teknik Regularisasi | MobileNetV2 | ResNet50 | Efek |
|-------------------|-------------|----------|------|
| **Dropout** | 0.3 + 0.2 | 0.5 | Mematikan neuron secara acak, cegah co-adaptation |
| **L2 Regularization** | 0.0003 | 0.001 | Membatasi magnitude weight, cegah overfitting |
| **Label Smoothing** | 0.05 | 0.10 | Mengurangi overconfidence pada prediksi |
| **Class Weights** | Tidak aktif | Aktif | Memberi bobot lebih pada kelas sulit |
| **Batch Normalization** | 2 layer | 1 layer | Stabilisasi distribusi internal |

**Mengapa ResNet50 butuh regularisasi lebih berat:**
- 25.6M parameter dengan 840 data → sangat rentan overfitting
- Dropout 0.5 (vs 0.3) mematikan lebih banyak neuron
- L2 = 0.001 (vs 0.0003) membatasi weight 3x lebih ketat
- Label smoothing 0.10 (vs 0.05) lebih kuat mencegah overconfidence

### 5.3 Faktor Strategi Training

| Aspek | MobileNetV2 | ResNet50 |
|-------|-------------|----------|
| **Training Strategy** | Two-Phase | Two-Phase |
| **Phase 1 LR** | 0.001 | 0.001 |
| **Phase 2 LR** | 3×10⁻⁵ | 5×10⁻⁵ |
| **Fine-tune Proportion** | 80/155 = 51.6% | 30/175 = 17.1% |
| **TTA Passes** | 5 | 5 |

**Mengapa Fine-tune Proportion berbeda:**
- MobileNetV2: Bisa fine-tune 51.6% layer karena model kecil, risiko overfitting rendah
- ResNet50: Hanya 17.1% layer yang di-fine-tune karena model besar, fine-tune terlalu banyak layer → overfitting

### 5.4 Faktor Preprocessing

| Model | Preprocessing | Range Pixel |
|-------|--------------|-------------|
| **ResNet50** | Caffe-style (subtract ImageNet mean) | ~[-123, 132] |
| **MobileNetV2** | TF-style (scale to [-1, 1]) | [-1, 1] |

Masing-masing model memerlukan preprocessing yang sesuai dengan cara model tersebut dilatih pada ImageNet. Penggunaan preprocessing yang salah dapat menurunkan performa signifikan.

---

## 6. Analisis Per-Kelas

### 6.1 Kelas "Baik" (Kerusakan Ringan/Tidak Ada)

| Metrik | ResNet50 | MobileNetV2 |
|--------|----------|-------------|
| Precision | 0.86 | 0.88 |
| Recall | 0.95 | 0.92 |
| F1-Score | 0.90 | 0.90 |

**Analisis:** Kedua model perform baik karena kelas "Baik" memiliki fitur visual yang paling jelas dan mudah dibedakan (permukaan jalan mulus, tanpa retakan).

### 6.2 Kelas "Berat" (Kerusakan Berat)

| Metrik | ResNet50 | MobileNetV2 |
|--------|----------|-------------|
| Precision | 0.78 | 0.80 |
| Recall | 0.80 | 0.85 |
| F1-Score | 0.79 | 0.82 |

**Analisis:** Kelas "Berat" memiliki fitur yang cukup distinguishable (lubang besar, retakan lebar). MobileNetV2 sedikit lebih baik karena depthwise separable convolution mampu menangkap pola tekstur dengan lebih efisien.

### 6.3 Kelas "Sedang" (Kerusakan Sedang) — Kelas Tersulit

| Metrik | ResNet50 | MobileNetV2 |
|--------|----------|-------------|
| Precision | 0.71 | 0.82 |
| Recall | **0.62** | **0.78** |
| F1-Score | 0.67 | 0.80 |

**Analisis:** Ini adalah kelas tersulit karena:
1. **Ambiguitas visual** — Kerusakan sedang bisa mirip dengan kerusakan ringan atau awal kerusakan berat
2. **Boundary tidak jelas** — Transisi antara "baik→sedang" dan "sedang→berat" bersifat gradual
3. **Variasi tinggi** — Kerusakan sedang memiliki banyak bentuk (retakan kecil, pengelupasan, deformasi ringan)

**Mengapa MobileNetV2 lebih baik pada kelas ini:**
- Model yang lebih kecil dengan regularisasi yang tepat mampu belajar **decision boundary yang lebih smooth** antara kelas
- ResNet50 yang overfitting cenderung membuat decision boundary yang terlalu tajam/kompleks, tidak generalize dengan baik

---

## 7. Dampak Jumlah Data — Analisis Detail

### 7.1 Skenario Ideal vs Aktual

| Skenario | Jumlah Data | Expected Accuracy |
|----------|-------------|-------------------|
| **Dataset saat ini** | 840 train (280/kelas) | 79-85% |
| **Dataset 2x** | 1.680 train (560/kelas) | 83-88% (estimasi) |
| **Dataset 5x** | 4.200 train (1.400/kelas) | 87-92% (estimasi) |
| **Dataset 10x** | 8.400 train (2.800/kelas) | 90-95% (estimasi) |

### 7.2 Mengapa Jumlah Data Sangat Berpengaruh?

#### Untuk MobileNetV2 (3.4M params):
- **840 data sudah cukup** untuk mencapai 85% karena model ringan
- Dengan 2.000+ data, kemungkinan bisa mencapai 90%+
- Augmentasi ringan sudah cukup efektif

#### Untuk ResNet50 (25.6M params):
- **840 data TIDAK CUKUP** — model terlalu besar untuk data sekecil ini
- Membutuhkan minimal **5.000+ gambar** per kelas untuk performa optimal
- Augmentasi kuat + regularisasi berat diperlukan sebagai kompensasi
- Bahkan dengan semua teknik anti-overfitting, ResNet50 masih kalah dari MobileNetV2

### 7.3 Hukum Skala (Scaling Law)

```
Performa Model ∝ f(Jumlah Data, Kapasitas Model, Kualitas Data)
```

- Jika **data << parameter model** → Overfitting (kasus ResNet50)
- Jika **data ≈ parameter model** → Optimal
- Jika **data >> parameter model** → Underfitting (perlu model lebih besar)

Pada dataset 840 gambar:
- MobileNetV2 berada di zona **mendekati optimal**
- ResNet50 berada di zona **severe overfitting**

---

## 8. Dampak Augmentasi — Analisis Detail

### 8.1 Efek Masing-Masing Teknik Augmentasi

| Teknik | Efek pada Gambar | Relevansi untuk Kerusakan Jalan |
|--------|-----------------|-------------------------------|
| **Rotation** | Memutar gambar | Kerusakan bisa difoto dari sudut berbeda |
| **Width/Height Shift** | Menggeser posisi | Kerusakan tidak selalu di tengah frame |
| **Shear** | Distorsi perspektif | Simulasi sudut kamera miring |
| **Zoom** | Memperbesar/perkecil | Kerusakan bisa difoto dari jarak berbeda |
| **Brightness** | Mengubah kecerahan | Simulasi kondisi pencahayaan berbeda (pagi/siang/mendung) |
| **Horizontal Flip** | Membalik horizontal | Kerusakan simetris, tidak terpengaruh orientasi |

### 8.2 Augmentasi Optimal untuk Dataset Kecil

#### Prinsip:
1. **Terlalu ringan** → Model tetap overfitting (gambar terlalu mirip aslinya)
2. **Terlalu agresif** → Model kesulitan belajar (gambar jadi tidak natural)
3. **Tepat** → Cukup variasi tanpa merusak informasi penting

#### Pengalaman pada Proyek Ini:

| Percobaan | Augmentasi | Hasil |
|-----------|-----------|-------|
| ResNet50 v1 (awal) | Agresif (rot=30, zoom=0.3) | Val 80%, Test 79% |
| MobileNetV2 v1 | Agresif + Mixup | Test hanya 78% (bahkan menurun!) |
| MobileNetV2 final | **Moderate** (rot=15, zoom=0.10) | **Test 85%** ✓ |
| ResNet50 v2 | Moderate (rot=15, zoom=0.10) | Overfitting parah (gap 15%) |
| ResNet50 v3 | Semi-agresif (rot=25, zoom=0.20) | Sedang diuji |

**Kesimpulan:**
- MobileNetV2: Cukup augmentasi **moderate** karena model sudah cukup compact
- ResNet50: Butuh augmentasi **lebih kuat** sebagai satu-satunya cara mengurangi overfitting tanpa menambah data

### 8.3 Test Time Augmentation (TTA)

TTA memberikan boost tambahan dengan cara:
1. Membuat **5 versi augmented** dari setiap gambar test
2. Menjalankan prediksi pada semua versi + versi asli (total 6 prediksi)
3. Mengambil **rata-rata probabilitas** dari semua prediksi
4. Prediksi final berdasarkan rata-rata tersebut

**Efek TTA:**
- MobileNetV2: Standard 83.33% → TTA **85.00%** (+1.67%)
- ResNet50: Standard 78.33% → TTA **79.17%** (+0.84%)

TTA lebih efektif pada model yang sudah baik (MobileNetV2) karena augmented predictions lebih konsisten.

---

## 9. Kesimpulan dan Rekomendasi

### 9.1 Mengapa MobileNetV2 Lebih Baik untuk Kasus Ini?

1. **Proporsionalitas model-data**: 3.4M parameter lebih cocok untuk 840 gambar training
2. **Efisiensi arsitektur**: Depthwise Separable Convolution mengekstrak fitur secara efisien
3. **Risiko overfitting rendah**: Lebih sedikit parameter = lebih sedikit yang bisa dihafal
4. **Fine-tune lebih banyak layer**: Bisa fine-tune 51.6% layer tanpa overfitting
5. **Regularisasi ringan sudah cukup**: Dropout 0.3 dan L2 0.0003 sudah efektif

### 9.2 Kapan ResNet50 Akan Lebih Baik?

ResNet50 akan mengungguli MobileNetV2 jika:
- Dataset diperbesar menjadi **5.000+ gambar per kelas**
- Masalah klasifikasi lebih kompleks (lebih dari 10 kelas)
- Variasi intra-kelas sangat tinggi
- Tersedia computing resource untuk training lebih lama

### 9.3 Rekomendasi untuk Pengembangan

| Prioritas | Aksi | Estimasi Dampak |
|-----------|------|-----------------|
| **Tinggi** | Tambah data training (minimal 500/kelas) | +3-5% accuracy |
| **Tinggi** | Perbaiki kualitas label kelas "Sedang" | +2-4% accuracy |
| **Sedang** | Gunakan ensemble (gabung prediksi kedua model) | +1-3% accuracy |
| **Sedang** | Coba EfficientNet-B0 (balance antara size & power) | Potensi > 85% |
| **Rendah** | Cross-validation untuk evaluasi lebih robust | Metrik lebih reliable |

---

## 10. Ringkasan Akhir

| Aspek | MobileNetV2 ✓ | ResNet50 |
|-------|---------------|----------|
| **Test Accuracy** | **85.00%** | 79.17% |
| **Ukuran Model** | ~30 MB | ~237 MB |
| **Kecepatan Inferensi** | Cepat | Lambat |
| **Cocok untuk Dataset Kecil** | **Ya** | Tidak |
| **Cocok untuk Mobile/Edge** | **Ya** | Tidak |
| **Overfitting Risk** | Rendah | Tinggi |
| **Kelas Tersulit (Sedang)** | F1=0.80 | F1=0.67 |

**Verdict:** Untuk dataset kerusakan jalan dengan 1.200 gambar (3 kelas), **MobileNetV2 adalah pilihan yang lebih tepat** karena keseimbangan antara kapasitas model dan ketersediaan data.

---

*Dokumen ini dibuat sebagai bagian dari penelitian klasifikasi kerusakan jalan menggunakan Convolutional Neural Network (CNN) dengan pendekatan Transfer Learning.*
