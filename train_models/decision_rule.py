# ============================================================
# DECISION RULE — Penanganan prediksi tie / near-tie
#
# Masalah: argmax selalu memaksa satu kelas, walaupun probabilitas
# hampir seimbang (mis. Sedang 50% : Berat 50%). Untuk kasus seperti
# ini sistem TIDAK memaksa salah satu kelas, tetapi mengembalikan
# status "AMBIGU — perlu verifikasi manual".
#
# Aturan (berlaku untuk jumlah kelas berapa pun):
#   p1 = probabilitas tertinggi, p2 = probabilitas tertinggi kedua
#   margin = p1 - p2
#
#   1. margin <  TIE_MARGIN       → AMBIGU (tie / near-tie)
#   2. p1     <  MIN_CONFIDENCE   → AMBIGU (confidence terlalu rendah)
#   3. selain itu                 → kelas = argmax (prediksi pasti)
#
# Contoh (TIE_MARGIN=0.10, MIN_CONFIDENCE=0.50):
#   [0.50, 0.50, 0.00] → margin 0.00 → AMBIGU (tie)
#   [0.47, 0.43, 0.10] → margin 0.04 → AMBIGU (near-tie)
#   [0.45, 0.30, 0.25] → margin 0.15 tapi p1<0.50 → AMBIGU (low confidence)
#   [0.70, 0.20, 0.10] → margin 0.50 → kelas pasti
# ============================================================

import numpy as np

TIE_MARGIN     = 0.10   # selisih top-1 vs top-2 minimal 10 poin persen
MIN_CONFIDENCE = 0.50   # top-1 minimal 50%

AMBIGUOUS_CLASS = 'ambigu'

REASON_TIE      = 'tie'
REASON_NEAR_TIE = 'near_tie'
REASON_LOW_CONF = 'low_confidence'

# Toleransi numerik: selisih <= TIE_EPS dianggap seri persis (mis. 0.5 vs 0.5)
TIE_EPS = 1e-6


def decide(probs, class_names, tie_margin: float = TIE_MARGIN,
           min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Terapkan aturan keputusan pada vektor probabilitas satu gambar.

    Returns dict:
        is_ambiguous : bool
        reason       : None | 'tie' | 'near_tie' | 'low_confidence'
        pred_class   : nama kelas final ('ambigu' jika ambigu)
        top1_class, top1_prob, top2_class, top2_prob, margin
        candidates   : daftar kelas kandidat yang harus diverifikasi
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.size != len(class_names):
        raise ValueError(f'Jumlah probabilitas ({probs.size}) != jumlah kelas ({len(class_names)})')

    order  = np.argsort(-probs, kind='stable')
    i1     = int(order[0])
    i2     = int(order[1]) if probs.size > 1 else i1
    p1     = float(probs[i1])
    p2     = float(probs[i2]) if probs.size > 1 else 0.0
    margin = p1 - p2

    reason = None
    if probs.size > 1 and margin <= TIE_EPS:
        reason = REASON_TIE
    elif probs.size > 1 and margin < tie_margin:
        reason = REASON_NEAR_TIE
    elif p1 < min_confidence:
        reason = REASON_LOW_CONF

    is_ambiguous = reason is not None

    if is_ambiguous:
        # Semua kelas yang berada dalam jarak TIE_MARGIN dari top-1 jadi kandidat
        candidates = [class_names[int(i)] for i in order if p1 - probs[int(i)] < max(tie_margin, TIE_EPS * 2)]
        if len(candidates) < 2 and probs.size > 1:
            candidates = [class_names[i1], class_names[i2]]
    else:
        candidates = [class_names[i1]]

    return {
        'is_ambiguous': is_ambiguous,
        'reason'      : reason,
        'pred_class'  : AMBIGUOUS_CLASS if is_ambiguous else class_names[i1],
        'top1_class'  : class_names[i1],
        'top1_prob'   : p1,
        'top2_class'  : class_names[i2],
        'top2_prob'   : p2,
        'margin'      : margin,
        'candidates'  : candidates,
    }


def selective_metrics(y_true, y_prob, tie_margin: float = TIE_MARGIN,
                      min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Evaluasi aturan ambigu pada data berlabel (mis. test set).

    Returns dict:
        n_total, n_ambiguous, coverage (proporsi diprediksi pasti),
        accuracy_all (argmax biasa, tanpa aturan),
        accuracy_accepted (akurasi pada prediksi yang TIDAK ambigu),
        error_rate_ambiguous (proporsi salah pada sampel ambigu bila dipaksa argmax)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)

    names = [str(i) for i in range(y_prob.shape[1])]
    ambiguous = np.array([
        decide(p, names, tie_margin, min_confidence)['is_ambiguous'] for p in y_prob
    ], dtype=bool)
    accepted = ~ambiguous
    correct  = y_pred == y_true

    n_total = int(y_true.size)
    n_amb   = int(ambiguous.sum())
    return {
        'tie_margin'          : float(tie_margin),
        'min_confidence'      : float(min_confidence),
        'n_total'             : n_total,
        'n_ambiguous'         : n_amb,
        'coverage'            : float(accepted.mean()) if n_total else float('nan'),
        'accuracy_all'        : float(correct.mean()) if n_total else float('nan'),
        'accuracy_accepted'   : float(correct[accepted].mean()) if accepted.any() else float('nan'),
        'error_rate_ambiguous': float((~correct[ambiguous]).mean()) if ambiguous.any() else float('nan'),
    }
