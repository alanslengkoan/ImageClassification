# ============================================================
# VALIDASI SEGFORMER-B0 — Ground-Truth Mask + mAP50-95
#
# Mengukur kualitas segmentasi jalan SegFormer-B0
# (Marco333/segformer-b0-road-scene-7class) yang dipakai di
# clean_objects.py dan aplikasi web, terhadap mask ground-truth
# hasil anotasi manual.
#
# Dua varian mask yang dievaluasi:
#   RAW  : output mentah SegFormer (argmax == road)
#   POST : mask setelah post-processing clean_objects.py
#          (morfologi + komponen bawah + fill holes + fallback)
#          → ini mask yang BENAR-BENAR dipakai sebelum klasifikasi.
#
# Metrik:
#   Pixel-level : IoU, Dice/F1, Precision, Recall, Pixel Accuracy, mIoU
#   Mask AP     : AP50, AP75, AP95, mAP50-95 (COCO, IoU 0.50:0.05:0.95)
#
# Struktur folder ground-truth:
#   train_models/segmentation_gt/
#   ├── images/<kelas>/<nama>.jpg       gambar asli (bukan yang sudah di-clean)
#   └── masks/<kelas>/<nama>.png        mask biner: putih(>0)=jalan, hitam=bukan jalan
#   (alternatif) images/<kelas>/<nama>.json  anotasi LabelMe (polygon label "jalan")
#
# Cara pakai:
#   1) Siapkan sampel gambar untuk dianotasi (stratified per kelas):
#        python3 validate_segformer.py --prepare 20
#   2) Anotasi area jalan (LabelMe / CVAT / GIMP) → simpan mask PNG / JSON LabelMe
#   3) Jalankan validasi:
#        python3 validate_segformer.py
# ============================================================

import os
import sys
import csv
import json
import time
import random
import argparse

import cv2
import numpy as np

from segmentation_metrics import (
    IOU_THRESHOLDS,
    aggregate_records,
    binary_segmentation_metrics,
    mask_confidence,
)

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR      = '/home/echolog/Documents/Project/www/skripsi/ImageClassification/train_models'
SOURCE_DIR    = os.path.join(BASE_DIR, 'dataset_all')               # gambar asli
GT_DIR        = os.path.join(BASE_DIR, 'segmentation_gt')
GT_IMAGE_DIR  = os.path.join(GT_DIR, 'images')
GT_MASK_DIR   = os.path.join(GT_DIR, 'masks')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'dataset', 'output', 'segmentation_validation')
MANIFEST_PATH = os.path.join(BASE_DIR, 'dataset', 'split_manifest.json')

CLASSES    = ['baik', 'sedang', 'berat']
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MASK_EXTS  = ('.png', '.bmp', '.tif', '.tiff')

# Label polygon LabelMe yang dianggap "jalan" (case-insensitive)
LABELME_ROAD_LABELS = {'jalan', 'road', 'aspal'}

SEED          = 42
PREVIEW_LIMIT = 30          # jumlah overlay preview yang disimpan
EVAL_MAX_SIDE = None        # None = resolusi asli (sama seperti clean_objects.py)


# ============================================================
# PERSIAPAN SAMPEL ANOTASI
# ============================================================
def prepare_samples(n_per_class: int):
    """Salin N gambar per kelas ke segmentation_gt/images/<kelas>/ untuk dianotasi.

    Jika split_manifest.json tersedia, sampel diambil dari TEST split agar
    validasi segmentasi memakai data yang sama dengan evaluasi klasifikasi.
    """
    import shutil

    rng = random.Random(SEED)
    manifest = None
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
        print(f'📄 Mengambil sampel dari TEST split: {MANIFEST_PATH}')
    else:
        print('⚠️  split_manifest.json tidak ada → sampel diambil acak dari dataset_all/')

    total = 0
    for cls in CLASSES:
        src = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(src):
            print(f'   ⚠️  Folder tidak ditemukan: {src}')
            continue

        if manifest and cls in manifest.get('classes', {}):
            pool = list(manifest['classes'][cls].get('test', []))
        else:
            pool = sorted(f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTS))
        pool = [f for f in pool if os.path.exists(os.path.join(src, f))]

        chosen = sorted(rng.sample(pool, min(n_per_class, len(pool))))
        dst = os.path.join(GT_IMAGE_DIR, cls)
        os.makedirs(dst, exist_ok=True)
        os.makedirs(os.path.join(GT_MASK_DIR, cls), exist_ok=True)
        for fname in chosen:
            shutil.copy(os.path.join(src, fname), os.path.join(dst, fname))
        total += len(chosen)
        print(f'   {cls:10s}: {len(chosen)} gambar → {dst}')

    print(f'\n✅ {total} gambar siap dianotasi di: {GT_IMAGE_DIR}')
    print('➡️  Anotasi area JALAN lalu simpan sebagai:')
    print(f'   - mask PNG biner  : {GT_MASK_DIR}/<kelas>/<nama>.png (putih=jalan), atau')
    print(f'   - JSON LabelMe    : di samping gambar, label polygon: {sorted(LABELME_ROAD_LABELS)}')
    print('➡️  Setelah itu jalankan: python3 validate_segformer.py')


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================
def load_labelme_mask(json_path: str, shape: tuple) -> np.ndarray:
    """Rasterisasi polygon LabelMe berlabel jalan → mask biner (0/1)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    h, w = shape[:2]
    src_h = int(data.get('imageHeight') or h)
    src_w = int(data.get('imageWidth') or w)
    sx, sy = w / float(src_w), h / float(src_h)

    mask = np.zeros((h, w), dtype=np.uint8)
    for shp in data.get('shapes', []):
        if str(shp.get('label', '')).strip().lower() not in LABELME_ROAD_LABELS:
            continue
        pts = np.array(shp.get('points', []), dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 3:
            continue
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def load_gt_mask(mask_path: str, shape: tuple) -> np.ndarray:
    """Load mask GT (PNG biner atau JSON LabelMe) dan samakan ukuran dengan gambar."""
    h, w = shape[:2]
    if mask_path.lower().endswith('.json'):
        return load_labelme_mask(mask_path, shape)

    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise ValueError(f'Mask tidak bisa dibaca: {mask_path}')
    if gt.shape != (h, w):
        # nearest agar label tetap biner (tidak ada nilai abu-abu baru)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    return (gt > 127).astype(np.uint8)


def find_gt_pairs() -> list:
    """Cari pasangan (gambar, mask GT) di segmentation_gt/."""
    pairs = []
    for cls in CLASSES:
        img_dir = os.path.join(GT_IMAGE_DIR, cls)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            mask_path = None
            for ext in MASK_EXTS:
                cand = os.path.join(GT_MASK_DIR, cls, stem + ext)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                for cand in (os.path.join(GT_MASK_DIR, cls, stem + '.json'),
                             os.path.join(img_dir, stem + '.json')):
                    if os.path.exists(cand):
                        mask_path = cand
                        break
            pairs.append({
                'class'    : cls,
                'filename' : fname,
                'image'    : os.path.join(img_dir, fname),
                'mask'     : mask_path,
            })
    return pairs


def _resize_for_eval(img: np.ndarray) -> np.ndarray:
    if EVAL_MAX_SIDE is None:
        return img
    h, w = img.shape[:2]
    scale = EVAL_MAX_SIDE / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


# ============================================================
# EVALUASI SATU GAMBAR
# ============================================================
def evaluate_pair(pair: dict) -> tuple:
    """Return (record_dict, img_bgr, gt_mask, raw_mask, post_mask)."""
    from clean_objects import KEEP_CLASS_IDS, predict_segformer, postprocess_road_mask

    img = cv2.imread(pair['image'])
    if img is None:
        raise ValueError(f'Gambar tidak bisa dibaca: {pair["image"]}')
    img = _resize_for_eval(img)
    gt  = load_gt_mask(pair['mask'], img.shape)

    started_at = time.time()
    class_map, road_prob = predict_segformer(img, return_prob=True)
    raw_mask = np.isin(class_map, KEEP_CLASS_IDS).astype(np.uint8)
    post_mask, used_fallback = postprocess_road_mask(class_map)
    infer_time = time.time() - started_at

    record = {
        'class'        : pair['class'],
        'filename'     : pair['filename'],
        'gt_ratio'     : float(gt.mean()),
        'gt_nonempty'  : bool(gt.any()),
        'used_fallback': bool(used_fallback),
        'time_sec'     : infer_time,
    }
    for prefix, pred in (('raw', raw_mask), ('post', post_mask)):
        m = binary_segmentation_metrics(pred, gt)
        for k, v in m.items():
            record[f'{prefix}_{k}'] = v
        record[f'{prefix}_pred_ratio']    = float(pred.mean())
        record[f'{prefix}_pred_nonempty'] = bool(pred.any())
        record[f'{prefix}_score']         = mask_confidence(road_prob, pred)

    return record, img, gt, raw_mask, post_mask


def save_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, record: dict, path: str):
    """Overlay: hijau=TP, merah=FP (prediksi jalan, GT bukan), biru=FN (terlewat)."""
    overlay = img.copy()
    gt_b, pr_b = gt.astype(bool), pred.astype(bool)
    overlay[pr_b & gt_b]  = (0, 200, 0)
    overlay[pr_b & ~gt_b] = (0, 0, 230)
    overlay[~pr_b & gt_b] = (230, 80, 0)
    vis = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

    gt_vis = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([img, gt_vis, vis])

    scale = 1600.0 / canvas.shape[1]
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    text = (f'{record["class"]} | IoU={record["post_iou"]:.3f} | Dice={record["post_dice"]:.3f} '
            f'| hijau=TP merah=FP biru=FN')
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


# ============================================================
# LAPORAN
# ============================================================
METRIC_ROWS = [
    ('IoU jalan (mean/gambar)', 'mean_iou'),
    ('Dice / F1 (mean/gambar)', 'mean_dice'),
    ('Precision (mean/gambar)', 'mean_precision'),
    ('Recall (mean/gambar)',    'mean_recall'),
    ('Pixel Accuracy (mean)',   'mean_pixel_acc'),
    ('mIoU jalan+latar (mean)', 'mean_miou'),
    ('IoU jalan (dataset)',     'micro_iou'),
    ('Dice / F1 (dataset)',     'micro_dice'),
    ('AP50',                    'ap50'),
    ('AP75',                    'ap75'),
    ('AP95',                    'ap95'),
    ('mAP50-95',                'map50_95'),
]


def _fmt(v) -> str:
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.4f}'


def build_report(records: list) -> tuple:
    """Return (summary_dict, report_text)."""
    summary = {
        'model'         : 'Marco333/segformer-b0-road-scene-7class',
        'num_images'    : len(records),
        'iou_thresholds': [float(t) for t in IOU_THRESHOLDS],
        'overall'       : {
            'raw' : aggregate_records(records, 'raw'),
            'post': aggregate_records(records, 'post'),
        },
        'per_class'     : {},
        'fallback_count': int(sum(r['used_fallback'] for r in records)),
        'mean_time_sec' : float(np.mean([r['time_sec'] for r in records])) if records else 0.0,
    }
    for cls in CLASSES:
        sub = [r for r in records if r['class'] == cls]
        if sub:
            summary['per_class'][cls] = {
                'raw' : aggregate_records(sub, 'raw'),
                'post': aggregate_records(sub, 'post'),
            }

    lines = []
    lines.append('Validasi Segmentasi — SegFormer-B0 vs Ground-Truth Mask')
    lines.append(f'Model      : {summary["model"]}')
    lines.append(f'Jumlah GT  : {len(records)} gambar')
    lines.append(f'Fallback   : {summary["fallback_count"]} gambar memakai fallback prior')
    lines.append(f'Waktu/img  : {summary["mean_time_sec"]:.2f} s')
    lines.append('RAW  = output mentah SegFormer (argmax == road)')
    lines.append('POST = mask setelah post-processing clean_objects.py (dipakai sebelum klasifikasi)')
    lines.append('=' * 64)
    lines.append(f'{"Metrik":<28} {"RAW":>10} {"POST":>10}')
    lines.append('-' * 64)
    raw, post = summary['overall']['raw'], summary['overall']['post']
    for label, key in METRIC_ROWS:
        lines.append(f'{label:<28} {_fmt(raw.get(key)):>10} {_fmt(post.get(key)):>10}')
    lines.append('-' * 64)
    lines.append('AP per threshold IoU (POST):')
    for thr, ap in post.get('ap_per_threshold', {}).items():
        lines.append(f'   AP@{thr}: {_fmt(ap)}')

    lines.append('')
    lines.append('Per kelas kondisi jalan (POST):')
    lines.append(f'{"Kelas":<10} {"N":>4} {"IoU":>8} {"Dice":>8} {"Prec":>8} {"Recall":>8} {"AP50":>8} {"mAP50-95":>9}')
    for cls, val in summary['per_class'].items():
        p = val['post']
        lines.append(f'{cls:<10} {p["num_images"]:>4} {_fmt(p["mean_iou"]):>8} {_fmt(p["mean_dice"]):>8} '
                     f'{_fmt(p["mean_precision"]):>8} {_fmt(p["mean_recall"]):>8} '
                     f'{_fmt(p["ap50"]):>8} {_fmt(p["map50_95"]):>9}')
    lines.append('=' * 64)
    lines.append('Catatan: mAP50-95 = rata-rata AP pada IoU 0.50, 0.55, ..., 0.95 (standar COCO).')
    lines.append('Satu gambar = satu instance "jalan"; skor = rata-rata probabilitas road di dalam mask.')
    return summary, '\n'.join(lines)


def save_csv(records: list, path: str):
    keys = list(records[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_plots(records: list, summary: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1) AP vs threshold IoU + distribusi IoU per gambar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for prefix, color in (('raw', '#FF9800'), ('post', '#2196F3')):
        ap_thr = summary['overall'][prefix].get('ap_per_threshold', {})
        xs = [float(k) for k in ap_thr.keys()]
        ys = [0.0 if np.isnan(v) else v for v in ap_thr.values()]
        axes[0].plot(xs, ys, marker='o', color=color, linewidth=2,
                     label=f'{prefix.upper()} (mAP50-95={_fmt(summary["overall"][prefix].get("map50_95"))})')
    axes[0].set_title('Mask AP vs Threshold IoU — SegFormer-B0', fontweight='bold')
    axes[0].set_xlabel('Threshold IoU'); axes[0].set_ylabel('AP')
    axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.3); axes[0].legend()

    bins = np.linspace(0, 1, 21)
    axes[1].hist([r['raw_iou'] for r in records], bins=bins, alpha=0.6,
                 color='#FF9800', label='RAW', edgecolor='black')
    axes[1].hist([r['post_iou'] for r in records], bins=bins, alpha=0.6,
                 color='#2196F3', label='POST', edgecolor='black')
    axes[1].set_title('Distribusi IoU Jalan per Gambar', fontweight='bold')
    axes[1].set_xlabel('IoU'); axes[1].set_ylabel('Jumlah gambar')
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'segformer_ap_iou.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Grafik AP & IoU tersimpan: {path}')

    # 2) Metrik per kelas kondisi jalan (POST)
    classes = list(summary['per_class'].keys())
    if not classes:
        return
    metrics = [('mean_iou', 'IoU'), ('mean_dice', 'Dice'), ('mean_precision', 'Precision'),
               ('mean_recall', 'Recall'), ('map50_95', 'mAP50-95')]
    x, width = np.arange(len(classes)), 0.16
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (key, label) in enumerate(metrics):
        vals = [summary['per_class'][c]['post'].get(key, 0.0) for c in classes]
        vals = [0.0 if np.isnan(v) else v for v in vals]
        bars = ax.bar(x + (i - 2) * width, vals, width, label=label, edgecolor='black')
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.title() for c in classes])
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('Metrik Segmentasi per Kelas Kondisi Jalan (POST) — SegFormer-B0', fontweight='bold')
    ax.legend(ncol=5, fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'segformer_per_class_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Grafik per kelas tersimpan: {path}')


# ============================================================
# MAIN
# ============================================================
def run_validation():
    pairs = find_gt_pairs()
    missing = [p for p in pairs if p['mask'] is None]
    pairs = [p for p in pairs if p['mask'] is not None]

    print(f'📂 Ground-truth folder : {GT_DIR}')
    print(f'   Pasangan gambar+mask : {len(pairs)}')
    if missing:
        print(f'   ⚠️  {len(missing)} gambar belum punya mask (dilewati), contoh: '
              f'{[m["filename"] for m in missing[:3]]}')
    if not pairs:
        print('\n❌ Tidak ada mask ground-truth.')
        print('   Jalankan dulu: python3 validate_segformer.py --prepare 20')
        print('   lalu anotasi area jalan dan simpan mask ke segmentation_gt/masks/<kelas>/')
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preview_dir = os.path.join(OUTPUT_DIR, 'preview')
    os.makedirs(preview_dir, exist_ok=True)

    records, errors = [], []
    for i, pair in enumerate(pairs, 1):
        try:
            record, img, gt, _, post_mask = evaluate_pair(pair)
        except Exception as e:
            errors.append(f'{pair["filename"]}: {e}')
            print(f'   ⚠️  Skip {pair["filename"]}: {e}')
            continue
        records.append(record)
        print(f'   [{i}/{len(pairs)}] {pair["class"]:7s} {pair["filename"][:40]:40s} '
              f'IoU raw={record["raw_iou"]:.3f} post={record["post_iou"]:.3f}')
        if len(records) <= PREVIEW_LIMIT:
            stem = os.path.splitext(pair['filename'])[0]
            save_overlay(img, gt, post_mask, record,
                         os.path.join(preview_dir, f'{pair["class"]}_{stem}.jpg'))

    if not records:
        print('❌ Semua gambar gagal dievaluasi.')
        sys.exit(1)

    summary, report = build_report(records)
    summary['errors'] = errors

    print('\n' + report)

    report_path = os.path.join(OUTPUT_DIR, 'segformer_validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(os.path.join(OUTPUT_DIR, 'segformer_validation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    save_csv(records, os.path.join(OUTPUT_DIR, 'segformer_validation_per_image.csv'))
    save_plots(records, summary)

    print(f'\n✅ Report  : {report_path}')
    print(f'📁 Semua hasil tersimpan di: {OUTPUT_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Validasi SegFormer-B0 dengan ground-truth mask + mAP50-95')
    parser.add_argument('--prepare', type=int, metavar='N',
                        help='salin N gambar/kelas ke segmentation_gt/images untuk dianotasi')
    args = parser.parse_args()

    if args.prepare:
        prepare_samples(args.prepare)
    else:
        run_validation()


if __name__ == '__main__':
    main()

