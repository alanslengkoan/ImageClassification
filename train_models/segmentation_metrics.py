# ============================================================
# SEGMENTATION METRICS — Validasi SegFormer-B0 vs Ground-Truth Mask
# Murni numpy (tanpa torch) agar mudah diuji & dipakai ulang.
#
# Metrik pixel-level (biner: jalan vs bukan jalan):
#   IoU (Jaccard), Dice (F1), Precision, Recall, Pixel Accuracy, mIoU
#
# Metrik mask-AP gaya COCO:
#   AP@0.50, AP@0.75, AP@0.95, mAP@[0.50:0.05:0.95]  (a.k.a. "mAP50-95")
#   - Setiap gambar menghasilkan maksimal 1 prediksi mask "jalan"
#     dengan skor confidence = rata-rata probabilitas jalan di dalam mask.
#   - Prediksi = True Positive jika IoU(pred, GT) >= threshold IoU.
#   - Prediksi diurutkan berdasarkan skor -> kurva Precision-Recall
#     -> AP dengan interpolasi 101 titik recall (standar COCO).
# ============================================================

import numpy as np

# 0.50, 0.55, ..., 0.95 (10 threshold, standar COCO)
IOU_THRESHOLDS = np.round(np.linspace(0.50, 0.95, 10), 2)
RECALL_POINTS  = np.linspace(0.0, 1.0, 101)


def _safe_div(num: float, den: float, empty_value: float = 1.0) -> float:
    """Pembagian aman. Jika penyebut 0 -> empty_value."""
    return float(num) / float(den) if den > 0 else float(empty_value)


# ============================================================
# PIXEL-LEVEL METRICS
# ============================================================
def confusion_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Hitung TP/FP/FN/TN piksel untuk mask biner (nilai != 0 = jalan)."""
    pred = np.asarray(pred_mask).astype(bool)
    gt   = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f'Ukuran mask berbeda: pred={pred.shape}, gt={gt.shape}')

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Metrik segmentasi biner dari confusion counts.

    Konvensi kasus kosong: jika GT dan prediksi sama-sama tidak memiliki
    piksel jalan, IoU/Dice/Precision/Recall = 1.0 (prediksi dianggap benar).
    """
    iou_road  = _safe_div(tp, tp + fp + fn)
    iou_bg    = _safe_div(tn, tn + fp + fn)
    dice      = _safe_div(2 * tp, 2 * tp + fp + fn)
    precision = _safe_div(tp, tp + fp, empty_value=1.0 if fn == 0 else 0.0)
    recall    = _safe_div(tp, tp + fn, empty_value=1.0 if fp == 0 else 0.0)
    pixel_acc = _safe_div(tp + tn, tp + fp + fn + tn)

    return {
        'iou'      : iou_road,
        'dice'     : dice,
        'precision': precision,
        'recall'   : recall,
        'pixel_acc': pixel_acc,
        'iou_bg'   : iou_bg,
        'miou'     : (iou_road + iou_bg) / 2.0,
    }


def binary_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Shortcut: confusion counts + metrik untuk satu pasang mask."""
    counts = confusion_counts(pred_mask, gt_mask)
    result = metrics_from_counts(**counts)
    result.update(counts)
    return result


def mask_confidence(prob_map: np.ndarray, pred_mask: np.ndarray) -> float:
    """Skor confidence mask = rata-rata probabilitas jalan di dalam mask prediksi."""
    pred = np.asarray(pred_mask).astype(bool)
    if not pred.any():
        return 0.0
    return float(np.asarray(prob_map, dtype=np.float64)[pred].mean())


# ============================================================
# MASK AVERAGE PRECISION (COCO-STYLE)
# ============================================================
def average_precision(scores, ious, pred_nonempty, gt_nonempty, iou_threshold: float) -> dict:
    """AP pada satu threshold IoU.

    Args:
        scores        : skor confidence prediksi per gambar
        ious          : IoU prediksi vs GT per gambar
        pred_nonempty : True jika model memprediksi ada jalan (ada deteksi)
        gt_nonempty   : True jika GT memiliki area jalan (ada objek GT)
        iou_threshold : batas IoU agar deteksi dihitung True Positive

    Returns:
        dict berisi 'ap', 'precision', 'recall' (kurva mentah) dan 'num_gt'.
        'ap' bernilai NaN bila tidak ada GT jalan sama sekali.
    """
    scores        = np.asarray(scores, dtype=np.float64)
    ious          = np.asarray(ious, dtype=np.float64)
    pred_nonempty = np.asarray(pred_nonempty, dtype=bool)
    gt_nonempty   = np.asarray(gt_nonempty, dtype=bool)

    num_gt = int(gt_nonempty.sum())
    if num_gt == 0:
        return {'ap': float('nan'), 'precision': np.array([]), 'recall': np.array([]), 'num_gt': 0}

    det_idx = np.flatnonzero(pred_nonempty)
    if det_idx.size == 0:
        return {'ap': 0.0, 'precision': np.array([]), 'recall': np.array([]), 'num_gt': num_gt}

    # Urutkan deteksi dari skor tertinggi (stable agar deterministik)
    order  = det_idx[np.argsort(-scores[det_idx], kind='stable')]
    is_tp  = gt_nonempty[order] & (ious[order] >= iou_threshold - 1e-12)
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)

    recall    = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # Precision envelope (monoton turun dari kanan) — sama seperti COCO
    envelope = precision.copy()
    for i in range(envelope.size - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])

    # Interpolasi 101 titik recall
    sampled = np.zeros_like(RECALL_POINTS)
    pos     = np.searchsorted(recall, RECALL_POINTS, side='left')
    valid   = pos < envelope.size
    sampled[valid] = envelope[pos[valid]]

    return {
        'ap'       : float(sampled.mean()),
        'precision': precision,
        'recall'   : recall,
        'num_gt'   : num_gt,
    }


def mean_average_precision(scores, ious, pred_nonempty, gt_nonempty,
                           thresholds=IOU_THRESHOLDS) -> dict:
    """AP per threshold IoU + mAP@[0.50:0.05:0.95] (mAP50-95)."""
    ap_per_thr = {}
    for thr in thresholds:
        thr = float(round(float(thr), 2))
        ap_per_thr[thr] = average_precision(scores, ious, pred_nonempty, gt_nonempty, thr)['ap']

    values  = np.array(list(ap_per_thr.values()), dtype=np.float64)
    map_all = float(np.nanmean(values)) if np.any(~np.isnan(values)) else float('nan')

    return {
        'ap_per_threshold': ap_per_thr,
        'ap50'    : ap_per_thr.get(0.5, float('nan')),
        'ap75'    : ap_per_thr.get(0.75, float('nan')),
        'ap95'    : ap_per_thr.get(0.95, float('nan')),
        'map50_95': map_all,
    }


# ============================================================
# AGREGASI DATASET
# ============================================================
def aggregate_records(records: list, prefix: str) -> dict:
    """Agregasi metrik untuk satu varian mask (mis. prefix='raw' atau 'post').

    Setiap record wajib memiliki key:
        f'{prefix}_tp', f'{prefix}_fp', f'{prefix}_fn', f'{prefix}_tn',
        f'{prefix}_iou', f'{prefix}_dice', f'{prefix}_precision',
        f'{prefix}_recall', f'{prefix}_pixel_acc', f'{prefix}_miou',
        f'{prefix}_score', f'{prefix}_pred_nonempty', 'gt_nonempty'

    Returns:
        dict berisi rata-rata per gambar (mean_*/std_*), metrik dataset-level
        (micro_*, dihitung dari total piksel), serta AP50/AP75/AP95/mAP50-95.
    """
    if len(records) == 0:
        return {'num_images': 0}

    def col(name):
        return np.array([r[f'{prefix}_{name}'] for r in records], dtype=np.float64)

    summary = {'num_images': len(records)}

    for name in ('iou', 'dice', 'precision', 'recall', 'pixel_acc', 'miou'):
        values = col(name)
        summary[f'mean_{name}'] = float(values.mean())
        summary[f'std_{name}']  = float(values.std())

    totals = {k: int(col(k).sum()) for k in ('tp', 'fp', 'fn', 'tn')}
    for name, value in metrics_from_counts(**totals).items():
        summary[f'micro_{name}'] = value

    ap = mean_average_precision(
        scores        = col('score'),
        ious          = col('iou'),
        pred_nonempty = np.array([r[f'{prefix}_pred_nonempty'] for r in records], dtype=bool),
        gt_nonempty   = np.array([r['gt_nonempty'] for r in records], dtype=bool),
    )
    summary.update({
        'ap50'            : ap['ap50'],
        'ap75'            : ap['ap75'],
        'ap95'            : ap['ap95'],
        'map50_95'        : ap['map50_95'],
        'ap_per_threshold': {f'{k:.2f}': v for k, v in ap['ap_per_threshold'].items()},
    })
    return summary

