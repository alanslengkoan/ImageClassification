# ============================================================
# CLEAN OBJECTS — Road Segmentation dengan SegFormer pre-trained
# Segmentasi pixel-level: keep hanya area jalan, mask sisanya
# Proses semua gambar di dataset_all/ → simpan ke dataset_all_clean/
# ============================================================

import os
import cv2
import numpy as np
from PIL import Image

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR     = '/home/echolog/Documents/Project/www/skripsi/ImageClassification/train_models'
SOURCE_DIR   = os.path.join(BASE_DIR, 'dataset_all')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'dataset_all_clean')

CLASSES      = ['baik', 'sedang', 'berat']

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Class mapping SegFormer road scene (7 class):
#   0=road, 1=sidewalk, 2=building, 3=vegetation,
#   4=sky, 5=vehicle, 6=roadside_object
# Kita keep hanya class 0 (road) — sisanya di-mask hitam
ROAD_CLASS_ID = 0
SIDEWALK_CLASS_ID = 1
MIN_ROAD_RATIO = 0.06

# ============================================================
# LOAD SEGFORMER PRE-TRAINED
# ============================================================
print('Loading SegFormer-B0 road scene segmentation...')
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import torch.nn.functional as F

processor = SegformerImageProcessor.from_pretrained(
    'Marco333/segformer-b0-road-scene-7class'
)
model = SegformerForSemanticSegmentation.from_pretrained(
    'Marco333/segformer-b0-road-scene-7class'
)
model.eval()
print('✅ SegFormer-B0 siap')

# ============================================================
# FUNGSI SEGMENTASI JALAN
# ============================================================
def segment_road(img_bgr: np.ndarray) -> np.ndarray:
    """Return gambar BGR dengan hanya area jalan, sisanya hitam."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    h, w = img_bgr.shape[:2]

    inputs = processor(images=pil_img, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits ke ukuran asli
    mask = F.interpolate(
        outputs.logits,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).argmax(dim=1)[0].numpy()

    # Buat mask: hanya pixel dengan class=road yang dipertahankan
    road_mask = (mask == ROAD_CLASS_ID).astype(np.uint8)  # 1=road, 0=non-road

    k = np.ones((7, 7), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k, iterations=1)
    road_mask = cv2.dilate(road_mask, k, iterations=1)

    h_mask, w_mask = road_mask.shape

    def _largest_bottom_component(bin_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num_labels <= 1:
            return bin_mask

        bottom_band = labels[max(0, h_mask - max(10, h_mask // 12)):h_mask, :]
        bottom_ids = np.unique(bottom_band)
        bottom_ids = bottom_ids[bottom_ids != 0]

        candidate_ids = bottom_ids if len(bottom_ids) > 0 else np.arange(1, num_labels)
        best_id = int(candidate_ids[np.argmax(stats[candidate_ids, cv2.CC_STAT_AREA])])
        return (labels == best_id).astype(np.uint8)

    road_mask = _largest_bottom_component(road_mask)
    road_ratio = float(road_mask.mean())

    if road_ratio < MIN_ROAD_RATIO:
        fallback = ((mask == ROAD_CLASS_ID) | (mask == SIDEWALK_CLASS_ID)).astype(np.uint8)

        prior = np.zeros_like(fallback, dtype=np.uint8)
        poly = np.array([
            [int(w_mask * 0.05), h_mask - 1],
            [int(w_mask * 0.95), h_mask - 1],
            [int(w_mask * 0.65), int(h_mask * 0.45)],
            [int(w_mask * 0.35), int(h_mask * 0.45)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(prior, poly, 1)

        fallback = fallback * prior
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, k, iterations=1)
        fallback = _largest_bottom_component(fallback)

        if float(fallback.mean()) >= MIN_ROAD_RATIO:
            road_mask = fallback
        else:
            road_mask = prior

    # Terapkan mask ke gambar asli
    result = img_bgr.copy()
    result[road_mask == 0] = 0  # non-road → hitam

    return result

# ============================================================
# PROSES SEMUA GAMBAR
# ============================================================
total_processed = 0
total_masked    = 0

for cls in CLASSES:
    src_path = os.path.join(SOURCE_DIR, cls)
    out_path = os.path.join(OUTPUT_DIR, cls)

    if not os.path.exists(src_path):
        print(f'⚠️  Folder tidak ditemukan: {src_path}')
        continue

    os.makedirs(out_path, exist_ok=True)

    images = sorted([f for f in os.listdir(src_path) if f.lower().endswith(IMAGE_EXTS)])
    n = len(images)
    print(f'\n📂 {cls}: {n} gambar')

    for i, fname in enumerate(images, 1):
        img_path = os.path.join(src_path, fname)
        save_path = os.path.join(out_path, fname)

        img = cv2.imread(img_path)
        if img is None:
            print(f'   ⚠️  Skip (tidak bisa dibaca): {fname}')
            continue

        # Segmentasi: keep hanya area jalan
        result = segment_road(img)

        cv2.imwrite(save_path, result)
        total_processed += 1

        # Cek apakah ada perubahan (ada area non-jalan yang di-mask)
        if not np.array_equal(img, result):
            total_masked += 1

        if i % 50 == 0 or i == n:
            print(f'   [{i}/{n}] {total_masked} gambar di-mask...')

print(f'\n{"=" * 55}')
print(f'🎉 Selesai!')
print(f'   Total diproses  : {total_processed} gambar')
print(f'   Total di-mask   : {total_masked} gambar')
print(f'   Output folder   : {OUTPUT_DIR}')
print(f'{"=" * 55}')
print(f'\n➡️  Selanjutnya: jalankan split_dataset.py')
print(f'   (pastikan SOURCE_DIR sudah diubah ke dataset_all_clean/)')
