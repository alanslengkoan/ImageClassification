import os, shutil, random

# ============================================================
# KONFIGURASI PATH & KELAS
# ============================================================
BASE_DIR   = '/home/echolog/Documents/Project/www/skripsi/ImageClassification-CNN/train_models'
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset_all')   # folder dengan 3 sub-folder: baik/sedang/berat
TRAIN_DIR  = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR    = os.path.join(BASE_DIR, 'dataset', 'val')
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')

CLASSES    = ['baik', 'sedang', 'berat']

TRAIN_RATIO = 0.7   # 70% train
VAL_RATIO   = 0.2   # 20% val
TEST_RATIO  = 0.1   # 10% test

random.seed(42)

# ============================================================
# CEK FOLDER SOURCE
# ============================================================
print('📂 Cek folder sumber:')
print('-' * 50)

if not os.path.exists(SOURCE_DIR):
    raise SystemExit(f'❌ Folder sumber tidak ditemukan: {SOURCE_DIR}')

total_source = 0
for cls in CLASSES:
    path = os.path.join(SOURCE_DIR, cls)
    if not os.path.exists(path):
        raise SystemExit(f'❌ Folder kelas tidak ditemukan: {path}')
    count = len([f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_source += count
    status = '✅' if count > 0 else '⚠️  kosong!'
    print(f'   {status} {cls:10s}: {count:3d} gambar')

print(f'\n   {"TOTAL":10s}: {total_source} gambar')

# ============================================================
# COLLECT GAMBAR PER KELAS
# ============================================================
print(f'\n🔍 Mengumpulkan dan mengacak gambar...')
images_per_class = {}
for cls in CLASSES:
    src  = os.path.join(SOURCE_DIR, cls)
    imgs = [f for f in os.listdir(src)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)
    images_per_class[cls] = imgs
    print(f'   {cls:10s}: {len(imgs):3d} gambar')

# ============================================================
# HAPUS FOLDER LAMA, BUAT ULANG
# ============================================================
print('\n🗑️  Menghapus folder train / val / test lama...')
for path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'   ✅ Dihapus : {path}')
    else:
        print(f'   ⚠️  Tidak ada: {path} (skip)')

print('\n📁 Membuat folder train / val / test baru...')
for cls in CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR,   cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR,  cls), exist_ok=True)
print(f'   ✅ Folder berhasil dibuat untuk kelas: {CLASSES}')

# ============================================================
# BAGI ACAK 70% TRAIN / 20% VAL / 10% TEST
# ============================================================
print('\n📊 Membagi dataset (70% train / 20% val / 10% test)...')
print('-' * 60)

total_train = 0
total_val   = 0
total_test  = 0

for cls in CLASSES:
    src        = os.path.join(SOURCE_DIR, cls)
    all_images = images_per_class[cls]

    n          = len(all_images)
    n_train    = int(n * TRAIN_RATIO)
    n_val      = int(n * VAL_RATIO)
    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:n_train + n_val]
    test_imgs  = all_images[n_train + n_val:]

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(TRAIN_DIR, cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(VAL_DIR, cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(TEST_DIR, cls, img))

    total_train += len(train_imgs)
    total_val   += len(val_imgs)
    total_test  += len(test_imgs)
    print(f'   {cls:10s}: {len(train_imgs):3d} train | {len(val_imgs):3d} val | {len(test_imgs):3d} test')

print('-' * 60)
print(f'   {"TOTAL":10s}: {total_train:3d} train | {total_val:3d} val | {total_test:3d} test')

# ============================================================
# VERIFIKASI HASIL
# ============================================================
print('\n✅ Verifikasi Hasil Pembagian:')
print('-' * 50)
for split, path in [('TRAIN', TRAIN_DIR), ('VAL', VAL_DIR), ('TEST', TEST_DIR)]:
    print(f'\n📁 {split}:')
    split_total = 0
    for cls in CLASSES:
        cls_path    = os.path.join(path, cls)
        count       = len(os.listdir(cls_path))
        split_total += count
        print(f'   {cls:10s}: {count} gambar')
    print(f'   {"TOTAL":10s}: {split_total} gambar')

# ============================================================
# RINGKASAN
# ============================================================
total_all = total_train + total_val + total_test
print(f'\n{"=" * 50}')
print(f'🎉 Dataset berhasil dibagi!')
print(f'   Kelas  : {CLASSES}')
print(f'   Train  : {total_train} gambar ({int(TRAIN_RATIO*100)}%)')
print(f'   Val    : {total_val} gambar ({int(VAL_RATIO*100)}%)')
print(f'   Test   : {total_test} gambar ({int(TEST_RATIO*100)}%)')
print(f'   Total  : {total_all} gambar')
print(f'{"=" * 50}')
print(f'\n📁 Struktur folder hasil:')
print(f'   {BASE_DIR}/dataset/')
print(f'   ├── train/')
for cls in CLASSES:
    print(f'   │   ├── {cls}/')
print(f'   ├── val/')
for cls in CLASSES:
    print(f'   │   ├── {cls}/')
print(f'   └── test/')
for cls in CLASSES:
    print(f'       ├── {cls}/')
print(f'\n➡️  Siap untuk training! Jalankan: python3 resnet50.py')