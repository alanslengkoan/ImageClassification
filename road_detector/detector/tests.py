"""Test aturan tie / near-tie pada web (SegFormer & TensorFlow di-mock).

Jalankan dari folder road_detector:
    python3 manage.py test detector
"""
import io
import sys
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from django.template.loader import render_to_string
from django.test import SimpleTestCase

from . import views

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'train_models'))
from decision_rule import decide  # noqa: E402

# CLASS_NAMES = ['baik', 'berat', 'sedang']


class _FakeModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)

    def predict(self, x, verbose=0):
        return self.probs[None, :]


class _FakeFile(io.BytesIO):
    name = 'jalan.jpg'


@mock.patch.object(views, '_save_cleaned_image', lambda img: '/media/cleaned/x.jpg')
@mock.patch.object(views, '_save_upload', lambda f: '/media/uploads/x.jpg')
@mock.patch.object(views, '_preprocess', lambda img: np.zeros((1, 224, 224, 3), np.float32))
@mock.patch.object(views, '_remove_objects', lambda f: Image.new('RGB', (32, 32)))
class TiePredictionTests(SimpleTestCase):

    def _predict(self, probs):
        with mock.patch('builtins.print'):
            return views._predict_one(_FakeFile(b''), _FakeModel(probs))

    def _render(self, result):
        return render_to_string('detector/index.html', {'results': [result], 'total': 1})

    def test_50_50_tidak_memaksa_kelas(self):
        r = self._predict([0.0, 0.5, 0.5])
        self.assertTrue(r['is_ambiguous'])
        self.assertTrue(r['is_tie'])
        self.assertEqual(r['pred_class'], 'ambigu')
        self.assertEqual(r['ambiguous_reason'], views.AMBIGUOUS_REASONS['tie'])
        self.assertCountEqual(r['candidates'], ['Berat', 'Sedang'])

        html = self._render(r)
        self.assertIn('Ambigu · Perlu Verifikasi', html)
        self.assertIn('Seri Berat = Sedang', html)
        self.assertNotIn('Top-1', html)

    def test_50_50_float32_dianggap_seri(self):
        r = self._predict([0.0, 0.49999997, 0.50000003])
        self.assertTrue(r['is_tie'])

    def test_seri_tiga_kelas(self):
        r = self._predict([1 / 3, 1 / 3, 1 / 3])
        self.assertTrue(r['is_tie'])
        self.assertEqual(len(r['candidates']), 3)

    def test_near_tie(self):
        r = self._predict([0.0, 0.48, 0.52])
        self.assertTrue(r['is_ambiguous'])
        self.assertFalse(r['is_tie'])
        self.assertEqual(r['ambiguous_reason'], views.AMBIGUOUS_REASONS['near_tie'])
        self.assertIn('Top-1 Sedang', self._render(r))

    def test_low_confidence(self):
        r = self._predict([0.45, 0.30, 0.25])
        self.assertTrue(r['is_ambiguous'])
        self.assertEqual(r['ambiguous_reason'], views.AMBIGUOUS_REASONS['low_confidence'])

    def test_prediksi_pasti(self):
        r = self._predict([0.70, 0.20, 0.10])
        self.assertFalse(r['is_ambiguous'])
        self.assertEqual(r['pred_class'], 'baik')
        self.assertNotIn('Perlu Verifikasi', self._render(r))


class DecideConsistencyTests(SimpleTestCase):
    """views._decide harus identik dengan train_models/decision_rule.decide."""

    def test_sama_dengan_decision_rule(self):
        rng = np.random.default_rng(0)
        vecs = [rng.dirichlet(np.ones(3) * a) for a in (0.3, 1, 5, 50) for _ in range(500)]
        vecs += [np.array(v) for v in ([0, .5, .5], [.5, 0, .5], [1 / 3] * 3, [.55, .45, 0])]
        for p in vecs:
            a = views._decide(p)
            b = decide(p, views.CLASS_NAMES)
            self.assertEqual(
                (a['is_ambiguous'], a['reason'], a['candidates'], views.CLASS_NAMES[a['top1_idx']]),
                (b['is_ambiguous'], b['reason'], b['candidates'], b['top1_class']),
                msg=f'probs={p}',
            )
