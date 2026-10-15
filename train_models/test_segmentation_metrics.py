# ============================================================
# UNIT TEST — segmentation_metrics.py & decision_rule.py
# Hanya butuh numpy (tanpa torch/tensorflow).
#
# Jalankan:  python3 test_segmentation_metrics.py
# ============================================================

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation_metrics import (
    binary_segmentation_metrics, average_precision, mean_average_precision,
    aggregate_records, mask_confidence,
)
from decision_rule import decide, selective_metrics

_passed = 0


def check(name, cond):
    global _passed
    assert cond, f'FAIL: {name}'
    _passed += 1
    print(f'PASS {name}')


def test_pixel_metrics():
    gt = np.zeros((10, 10), np.uint8); gt[5:, :] = 1          # 50 piksel jalan
    pred = np.zeros((10, 10), np.uint8); pred[6:, :] = 1      # 40 piksel, semua di dalam GT
    m = binary_segmentation_metrics(pred, gt)
    check('tp/fp/fn/tn', (m['tp'], m['fp'], m['fn'], m['tn']) == (40, 0, 10, 50))
    check('iou = 0.8', abs(m['iou'] - 0.8) < 1e-9)
    check('dice = 80/90', abs(m['dice'] - 80 / 90) < 1e-9)
    check('precision = 1', m['precision'] == 1.0)
    check('recall = 0.8', abs(m['recall'] - 0.8) < 1e-9)
    check('pixel acc = 0.9', abs(m['pixel_acc'] - 0.9) < 1e-9)
    check('miou jalan+latar', abs(m['miou'] - (0.8 + 50 / 60) / 2) < 1e-9)

    e = binary_segmentation_metrics(np.zeros((4, 4)), np.zeros((4, 4)))
    check('pred & GT kosong -> 1', e['iou'] == 1.0 and e['dice'] == 1.0)
    e = binary_segmentation_metrics(np.ones((4, 4)), np.zeros((4, 4)))
    check('hanya prediksi -> 0', e['iou'] == 0.0 and e['precision'] == 0.0)


def test_average_precision():
    r = mean_average_precision([0.9, 0.8, 0.7], [1.0, 1.0, 1.0], [1, 1, 1], [1, 1, 1])
    check('prediksi sempurna -> mAP50-95 = 1', abs(r['map50_95'] - 1.0) < 1e-9)

    # IoU [0.97, 0.80, 0.60]: thr 0.50-0.60 -> 3 TP, 0.65-0.80 -> 2 TP, 0.85-0.95 -> 1 TP
    r = mean_average_precision([0.9, 0.8, 0.7], [0.97, 0.80, 0.60], [1, 1, 1], [1, 1, 1])
    check('AP50 = 1', abs(r['ap50'] - 1.0) < 1e-9)
    check('AP75 = 67/101', abs(r['ap75'] - 67 / 101) < 1e-9)
    check('AP95 = 34/101', abs(r['ap95'] - 34 / 101) < 1e-9)
    expected = np.mean([1.0] * 3 + [67 / 101] * 4 + [34 / 101] * 3)
    check('mAP50-95', abs(r['map50_95'] - expected) < 1e-9)

    a = average_precision([0.9, 0.1], [0.2, 0.9], [1, 1], [1, 1], 0.5)['ap']
    b = average_precision([0.1, 0.9], [0.2, 0.9], [1, 1], [1, 1], 0.5)['ap']
    check('urutan skor berpengaruh', b > a)

    check('tanpa GT -> NaN', np.isnan(average_precision([0.9], [0.0], [1], [0], 0.5)['ap']))
    check('FP pada GT kosong menurunkan AP',
          average_precision([0.9, 0.8], [0.0, 0.9], [1, 1], [0, 1], 0.5)['ap'] < 1.0)
    check('jalan tidak terdeteksi -> AP 0', average_precision([0.0], [0.0], [0], [1], 0.5)['ap'] == 0.0)
    check('mask confidence', abs(mask_confidence(np.full((2, 2), 0.7), np.ones((2, 2))) - 0.7) < 1e-9)

    recs = []
    for i, iou in enumerate([0.97, 0.80, 0.60]):
        rec = {'gt_nonempty': True}
        for k, v in dict(tp=10, fp=1, fn=1, tn=88, iou=iou, dice=0.9, precision=0.9, recall=0.9,
                         pixel_acc=0.98, miou=0.9, score=0.9 - i * 0.1, pred_nonempty=True).items():
            rec[f'post_{k}'] = v
        recs.append(rec)
    s = aggregate_records(recs, 'post')
    check('aggregate mAP50-95', abs(s['map50_95'] - expected) < 1e-9)
    check('aggregate micro IoU', abs(s['micro_iou'] - 30 / 36) < 1e-9)


def test_decision_rule():
    names = ['baik', 'berat', 'sedang']
    d = decide([0.0, 0.5, 0.5], names)
    check('50:50 -> ambigu (tie)', d['is_ambiguous'] and d['reason'] == 'tie' and d['pred_class'] == 'ambigu')
    check('kandidat tie = berat/sedang', set(d['candidates']) == {'berat', 'sedang'})
    d = decide([0.10, 0.47, 0.43], names)
    check('near-tie', d['is_ambiguous'] and d['reason'] == 'near_tie')
    d = decide([0.45, 0.30, 0.25], names)
    check('low confidence', d['is_ambiguous'] and d['reason'] == 'low_confidence')
    d = decide([0.70, 0.20, 0.10], names)
    check('prediksi pasti', not d['is_ambiguous'] and d['pred_class'] == 'baik')
    d = decide([1 / 3, 1 / 3, 1 / 3], names)
    check('seri tiga kelas', d['is_ambiguous'] and len(d['candidates']) == 3)
    d = decide([0.55, 0.45, 0.0], names)
    check('margin tepat 0.10 -> bukan near-tie', not d['is_ambiguous'])

    sm = selective_metrics([0, 1, 2, 1], [[0.9, 0.05, 0.05], [0.0, 0.5, 0.5],
                                          [0.1, 0.2, 0.7], [0.1, 0.48, 0.42]])
    check('selective coverage', sm['n_ambiguous'] == 2 and abs(sm['coverage'] - 0.5) < 1e-9)
    check('akurasi prediksi pasti', sm['accuracy_accepted'] == 1.0)


if __name__ == '__main__':
    test_pixel_metrics()
    test_average_precision()
    test_decision_rule()
    print(f'\n✅ ALL {_passed} TESTS PASSED')
