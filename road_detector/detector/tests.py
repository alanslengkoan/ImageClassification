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
        self.assertIn('bg-gray-600', html)                             # badge ambigu
        self.assertIn('Prediksi ambigu / perlu verifikasi manual', html)
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
        html = self._render(r)
        self.assertNotIn('bg-gray-600', html)                          # badge ambigu
        self.assertNotIn('Prediksi ambigu / perlu verifikasi manual', html)
        # data-width harus memakai titik desimal (CSS), bukan koma locale 'id'
        self.assertIn('data-width="70.0"', html)
        self.assertNotIn('data-width="70,0"', html)


@mock.patch.object(views, '_save_cleaned_image', lambda img: '/media/cleaned/x.jpg')
@mock.patch.object(views, '_save_upload', lambda f: '/media/uploads/x.jpg')
@mock.patch.object(views, '_preprocess', lambda img: np.zeros((1, 224, 224, 3), np.float32))
@mock.patch.object(views, '_remove_objects', lambda f: Image.new('RGB', (32, 32)))
class ConfidenceNoteTests(SimpleTestCase):
    """Keterangan confidence di website harus konsisten dengan aturan ambigu."""

    def _predict(self, probs):
        with mock.patch('builtins.print'):
            return views._predict_one(_FakeFile(b''), _FakeModel(probs))

    def _render(self, result):
        return render_to_string('detector/index.html', {
            'results': [result], 'total': 1,
            'min_confidence_pct': result['min_confidence_pct'],
            'tie_margin_pct': result['tie_margin_pct'],
        })

    def test_keterangan_selalu_tampil(self):
        for probs in ([0.0, 0.5, 0.5], [0.70, 0.20, 0.10], [0.45, 0.30, 0.25]):
            html = self._render(self._predict(probs))
            self.assertIn('Keterangan confidence', html)
            self.assertIn('Cara membaca nilai confidence', html)
            self.assertIn('Confidence ·', html)
            self.assertIn('Probabilitas tiap kelas (total 100%)', html)

    def test_syarat_konsisten_dengan_status_ambigu(self):
        rng = np.random.default_rng(1)
        vecs = [rng.dirichlet(np.ones(3) * a) for a in (0.3, 1, 5, 50) for _ in range(100)]
        vecs += [[0.0, 0.5, 0.5], [0.55, 0.45, 0.0], [0.5, 0.39, 0.11], [1 / 3] * 3]
        for p in vecs:
            r = self._predict(p)
            self.assertEqual(r['is_ambiguous'], not (r['conf_ok'] and r['margin_ok']), msg=f'probs={p}')

    def test_keterangan_50_50(self):
        r = self._predict([0.0, 0.5, 0.5])
        self.assertTrue(r['conf_ok'])        # 50% tepat di batas minimum
        self.assertFalse(r['margin_ok'])     # selisih 0 poin
        html = self._render(r)
        self.assertIn('sama dengan', html)
        self.assertIn('Model tidak dapat membedakan Berat dan Sedang', html)
        self.assertNotIn('Keyakinan model sedang', html)
        self.assertIn('Hasil ambigu, perlu verifikasi manual', html)

    def test_keterangan_near_tie(self):
        html = self._render(self._predict([0.0, 0.48, 0.52]))
        self.assertIn('Model ragu di antara Sedang dan Berat', html)

    def test_keterangan_prediksi_pasti(self):
        r = self._predict([0.95, 0.03, 0.02])
        html = self._render(r)
        self.assertIn('Model sangat yakin', html)
        self.assertIn('prediksi dianggap pasti: Baik', html)
        self.assertNotIn('❌', html)

    def test_ambang_mengikuti_settings(self):
        with self.settings(PREDICTION_TIE_MARGIN=0.20, PREDICTION_MIN_CONFIDENCE=0.60):
            r = self._predict([0.65, 0.30, 0.05])
            self.assertEqual((r['min_confidence_pct'], r['tie_margin_pct']), (60.0, 20.0))
            self.assertFalse(r['is_ambiguous'])
            r = self._predict([0.55, 0.40, 0.05])
            self.assertTrue(r['is_ambiguous'])
            self.assertFalse(r['conf_ok'])


@mock.patch.object(views, '_save_cleaned_image', lambda img: '/media/cleaned/x.jpg')
@mock.patch.object(views, '_save_upload', lambda f: '/media/uploads/x.jpg')
@mock.patch.object(views, '_preprocess', lambda img: np.zeros((1, 224, 224, 3), np.float32))
@mock.patch.object(views, '_remove_objects', lambda f: Image.new('RGB', (32, 32)))
@mock.patch.object(views, '_get_segformer', lambda: (None, None))
class IndexViewTests(SimpleTestCase):
    """Upload lewat halaman utama (POST) menampilkan panduan & keterangan confidence."""

    def test_upload_50_50(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        upload = SimpleUploadedFile('jalan.jpg', b'x', content_type='image/jpeg')
        with mock.patch.object(views, '_get_model', lambda: _FakeModel([0.0, 0.5, 0.5])), \
             mock.patch('builtins.print'):
            resp = self.client.post('/', {'images': [upload]})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['min_confidence_pct'], 50.0)
        self.assertEqual(resp.context['tie_margin_pct'], 10.0)
        html = resp.content.decode()
        self.assertIn('Cara membaca nilai confidence', html)
        self.assertIn('Keterangan confidence', html)
        self.assertIn('Seri Berat = Sedang', html)
        self.assertIn('Hasil ambigu, perlu verifikasi manual', html)


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
