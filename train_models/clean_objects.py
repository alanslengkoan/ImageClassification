# ============================================================
# CLEAN OBJECTS — Road Segmentation dengan SegFormer pre-trained
# Segmentasi pixel-level: keep area jalan, redupkan sisanya
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
# Fokus utama ke kelas jalan agar remove object terlihat rapi
ROAD_CLASS_ID = 0
SIDEWALK_CLASS_ID = 1
ROADSIDE_OBJECT_CLASS_ID = 6
KEEP_SIDEWALK = False
KEEP_ROADSIDE_OBJECT = False

KEEP_CLASS_IDS = [ROAD_CLASS_ID]
if KEEP_SIDEWALK:
    KEEP_CLASS_IDS.append(SIDEWALK_CLASS_ID)
if KEEP_ROADSIDE_OBJECT:
    KEEP_CLASS_IDS.append(ROADSIDE_OBJECT_CLASS_ID)

MIN_ROAD_RATIO = 0.03
BACKGROUND_KEEP_ALPHA = 0.06
BACKGROUND_BLUR_SIGMA = 12
EDGE_FEATHER_SIGMA = 3.5

SAVE_PREVIEW = True
PREVIEW_LIMIT_PER_CLASS = 25

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
def segment_road(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Return (result_bgr, road_mask_255) dengan fokus utama pada area jalan."""
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

    # Buat mask awal sesuai class yang dipilih (default: road only)
    road_mask = np.isin(mask, KEEP_CLASS_IDS).astype(np.uint8)  # 1=keep, 0=mask

    k_close = np.ones((7, 7), np.uint8)
    k_open = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    h_mask, w_mask = road_mask.shape

    def _bottom_connected_components(bin_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num_labels <= 1:
            return bin_mask

        bottom_band = labels[max(0, h_mask - max(10, h_mask // 12)):h_mask, :]
        bottom_ids = np.unique(bottom_band)
        bottom_ids = bottom_ids[bottom_ids != 0]

        if len(bottom_ids) == 0:
            candidate_ids = np.arange(1, num_labels)
            best_id = int(candidate_ids[np.argmax(stats[candidate_ids, cv2.CC_STAT_AREA])])
            return (labels == best_id).astype(np.uint8)

        keep = np.zeros_like(bin_mask, dtype=np.uint8)
        min_area = max(200, int(0.002 * h_mask * w_mask))
        for comp_id in bottom_ids:
            if stats[comp_id, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == comp_id] = 1

        if keep.sum() == 0:
            keep = (labels == int(bottom_ids[0])).astype(np.uint8)

        return keep

    def _fill_holes(bin_mask: np.ndarray) -> np.ndarray:
        flood = bin_mask.copy().astype(np.uint8)
        h_f, w_f = flood.shape
        flood_canvas = np.zeros((h_f + 2, w_f + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_canvas, (0, 0), 1)
        holes = (1 - flood) & (1 - bin_mask)
        return (bin_mask | holes).astype(np.uint8)

    road_mask = _bottom_connected_components(road_mask)
    road_mask = _fill_holes(road_mask)
    road_ratio = float(road_mask.mean())
    used_fallback = False

    if road_ratio < MIN_ROAD_RATIO:
        used_fallback = True
        fallback = np.isin(mask, KEEP_CLASS_IDS).astype(np.uint8)

        prior = np.zeros_like(fallback, dtype=np.uint8)
        poly = np.array([
            [int(w_mask * 0.03), h_mask - 1],
            [int(w_mask * 0.97), h_mask - 1],
            [int(w_mask * 0.70), int(h_mask * 0.35)],
            [int(w_mask * 0.30), int(h_mask * 0.35)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(prior, poly, 1)

        fallback = fallback * prior
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, k_close, iterations=1)
        fallback = _fill_holes(fallback)
        fallback = _bottom_connected_components(fallback)

        if float(fallback.mean()) >= MIN_ROAD_RATIO:
            road_mask = fallback
        else:
            road_mask = prior

    # Feather edge agar transisi road vs non-road lebih natural/rapi
    soft_mask = cv2.GaussianBlur(road_mask.astype(np.float32), (0, 0), EDGE_FEATHER_SIGMA)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., None]

    # Background dibuat blur + redup agar fokus visual ke jalan
    fg = img_bgr.astype(np.float32)
    bg_blur = cv2.GaussianBlur(img_bgr, (0, 0), BACKGROUND_BLUR_SIGMA).astype(np.float32)
    bg = bg_blur * BACKGROUND_KEEP_ALPHA

    result = (fg * soft_mask) + (bg * (1.0 - soft_mask))
    result = np.clip(result, 0, 255).astype(np.uint8)

    road_mask_u8 = (road_mask * 255).astype(np.uint8)
    final_road_ratio = float(road_mask.mean())
    return result, road_mask_u8, final_road_ratio, used_fallback

# ============================================================
# PROSES SEMUA GAMBAR
# ============================================================
total_processed = 0
total_masked    = 0
total_fallback  = 0
all_road_ratios = []

for cls in CLASSES:
    src_path = os.path.join(SOURCE_DIR, cls)
    out_path = os.path.join(OUTPUT_DIR, cls)
    preview_path = os.path.join(OUTPUT_DIR, '_preview', cls)

    if not os.path.exists(src_path):
        print(f'⚠️  Folder tidak ditemukan: {src_path}')
        continue

    os.makedirs(out_path, exist_ok=True)
    if SAVE_PREVIEW:
        os.makedirs(preview_path, exist_ok=True)

    images = sorted([f for f in os.listdir(src_path) if f.lower().endswith(IMAGE_EXTS)])
    n = len(images)
    print(f'\n📂 {cls}: {n} gambar')
    cls_road_ratios = []
    cls_fallback_count = 0

    for i, fname in enumerate(images, 1):
        img_path = os.path.join(src_path, fname)
        save_path = os.path.join(out_path, fname)

        img = cv2.imread(img_path)
        if img is None:
            print(f'   ⚠️  Skip (tidak bisa dibaca): {fname}')
            continue

        # Segmentasi: keep fokus jalan, redupkan area non-jalan
        result, road_mask, road_ratio, used_fallback = segment_road(img)

        cv2.imwrite(save_path, result)
        total_processed += 1
        cls_road_ratios.append(road_ratio)
        all_road_ratios.append(road_ratio)
        if used_fallback:
            cls_fallback_count += 1
            total_fallback += 1

        # Cek apakah ada perubahan (ada area non-jalan yang di-mask)
        if not np.array_equal(img, result):
            total_masked += 1

        if SAVE_PREVIEW and i <= PREVIEW_LIMIT_PER_CLASS:
            mask_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            comparison = np.hstack([img, mask_vis, result])
            preview_file = os.path.join(preview_path, f'preview_{i:04d}_{fname}')
            cv2.imwrite(preview_file, comparison)

        if i % 50 == 0 or i == n:
            print(f'   [{i}/{n}] {total_masked} gambar di-mask...')

    if len(cls_road_ratios) > 0:
        cls_min = float(np.min(cls_road_ratios))
        cls_med = float(np.median(cls_road_ratios))
        cls_max = float(np.max(cls_road_ratios))
        cls_mean = float(np.mean(cls_road_ratios))
        print(f'   Road ratio {cls:10s}: min={cls_min:.3f} | median={cls_med:.3f} | mean={cls_mean:.3f} | max={cls_max:.3f}')
        print(f'   Fallback digunakan : {cls_fallback_count}/{len(cls_road_ratios)} gambar')

print(f'\n{"=" * 55}')
print(f'🎉 Selesai!')
print(f'   Total diproses  : {total_processed} gambar')
print(f'   Total di-mask   : {total_masked} gambar')
print(f'   Total fallback  : {total_fallback} gambar')
if len(all_road_ratios) > 0:
    all_min = float(np.min(all_road_ratios))
    all_med = float(np.median(all_road_ratios))
    all_max = float(np.max(all_road_ratios))
    all_mean = float(np.mean(all_road_ratios))
    print(f'   Road ratio all  : min={all_min:.3f} | median={all_med:.3f} | mean={all_mean:.3f} | max={all_max:.3f}')
print(f'   Output folder   : {OUTPUT_DIR}')
if SAVE_PREVIEW:
    print(f'   Preview folder  : {os.path.join(OUTPUT_DIR, "_preview")}')
print(f'{"=" * 55}')
print(f'\n➡️  Selanjutnya: jalankan split_dataset.py')
print(f'   (pastikan SOURCE_DIR sudah diubah ke dataset_all_clean/)')
