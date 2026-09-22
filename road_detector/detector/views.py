import os
import uuid
import numpy as np

from django.conf import settings
from django.shortcuts import render

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Model di-load sekali saat pertama kali dipakai (lazy load)
_model = None
_segformer = None
_seg_processor = None

ROAD_CLASS_ID = 0
SIDEWALK_CLASS_ID = 1
MIN_ROAD_RATIO = 0.03

CLASS_NAMES = ['baik', 'berat', 'sedang']   # urutan alphabetical ImageDataGenerator

CLASS_INFO = {
    'baik'  : {'label': 'Baik',   'color': 'green',  'icon': '🟢',
               'desc': 'Kondisi jalan baik, tidak terdeteksi kerusakan signifikan.'},
    'sedang': {'label': 'Sedang', 'color': 'yellow', 'icon': '🟡',
               'desc': 'Terdapat kerusakan ringan hingga sedang pada permukaan jalan.'},
    'berat' : {'label': 'Berat',  'color': 'red',    'icon': '🔴',
               'desc': 'Kerusakan berat terdeteksi, perlu perbaikan segera.'},
}


def _get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load

        model_path = str(settings.MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'Model tidak ditemukan: {model_path}\n'
                'Jalankan resnet50.py terlebih dahulu untuk melatih model.'
            )
        _model = keras_load(model_path)
    return _model


def _get_segformer():
    global _segformer, _seg_processor
    if _segformer is None:
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        _seg_processor = SegformerImageProcessor.from_pretrained(
            'Marco333/segformer-b0-road-scene-7class'
        )
        _segformer = SegformerForSemanticSegmentation.from_pretrained(
            'Marco333/segformer-b0-road-scene-7class'
        )
        _segformer.eval()
    return _segformer, _seg_processor


def _remove_objects(image_file) -> Image.Image:
    import cv2
    import torch
    import torch.nn.functional as F

    model, processor = _get_segformer()

    img = Image.open(image_file).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    inputs = processor(images=img, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    mask = F.interpolate(
        outputs.logits,
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).argmax(dim=1)[0].numpy()

    road_mask = (mask == ROAD_CLASS_ID).astype(np.uint8)

    k_close = np.ones((7, 7), np.uint8)
    k_open = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k_open, iterations=1)

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

    def _fill_holes(bin_mask: np.ndarray) -> np.ndarray:
        flood = bin_mask.copy().astype(np.uint8)
        flood_canvas = np.zeros((h_mask + 2, w_mask + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_canvas, (0, 0), 1)
        holes = (1 - flood) & (1 - bin_mask)
        return (bin_mask | holes).astype(np.uint8)

    road_mask = _largest_bottom_component(road_mask)
    road_mask = _fill_holes(road_mask)
    road_ratio = float(road_mask.mean())

    if road_ratio < MIN_ROAD_RATIO:
        fallback = ((mask == ROAD_CLASS_ID) | (mask == SIDEWALK_CLASS_ID)).astype(np.uint8)

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
        fallback = _largest_bottom_component(fallback)

        if float(fallback.mean()) >= MIN_ROAD_RATIO:
            road_mask = fallback
        else:
            road_mask = prior

    soft_mask = cv2.GaussianBlur(road_mask.astype(np.float32), (0, 0), 3.5)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., None]

    fg = img_cv.astype(np.float32)
    bg_blur = cv2.GaussianBlur(img_cv, (0, 0), 12).astype(np.float32)
    bg = bg_blur * 0.06

    result = (fg * soft_mask) + (bg * (1.0 - soft_mask))
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def _preprocess(img: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def _save_upload(image_file) -> str:
    ext      = os.path.splitext(image_file.name)[-1].lower() or '.jpg'
    filename = f'{uuid.uuid4().hex}{ext}'
    save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'wb') as f:
        for chunk in image_file.chunks():
            f.write(chunk)
    return f'{settings.MEDIA_URL}uploads/{filename}'


def _predict_one(image_file, model):
    image_url = _save_upload(image_file)
    image_file.seek(0)

    img_clean = _remove_objects(image_file)
    image_file.seek(0)

    tensor = _preprocess(img_clean)
    probs  = model.predict(tensor, verbose=0)[0]

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    info       = CLASS_INFO[pred_class]

    probabilities = [
        {
            'class': cls,
            'label': CLASS_INFO[cls]['label'],
            'color': CLASS_INFO[cls]['color'],
            'value': round(float(probs[i]) * 100, 2),
            'width': round(float(probs[i]) * 100, 1),
        }
        for i, cls in enumerate(CLASS_NAMES)
    ]
    probabilities.sort(key=lambda x: x['value'], reverse=True)

    return {
        'image_url'    : image_url,
        'filename'     : image_file.name,
        'pred_class'   : pred_class,
        'pred_label'   : info['label'],
        'pred_color'   : info['color'],
        'pred_icon'    : info['icon'],
        'pred_desc'    : info['desc'],
        'confidence'   : round(float(probs[pred_idx]) * 100, 2),
        'probabilities': probabilities,
    }


def index(request):
    context = {}

    if request.method == 'POST':
        files = request.FILES.getlist('images')

        if not files:
            context['error'] = 'Pilih minimal satu gambar untuk dianalisis.'
            return render(request, 'detector/index.html', context)

        max_files = getattr(settings, 'MAX_UPLOAD_FILES', 10)
        if len(files) > max_files:
            context['error'] = f'Maksimal {max_files} gambar sekaligus.'
            return render(request, 'detector/index.html', context)

        try:
            model   = _get_model()
            results = []
            errors  = []

            for f in files:
                try:
                    results.append(_predict_one(f, model))
                except Exception as e:
                    errors.append(f'{f.name}: {e}')

            context = {
                'results'     : results,
                'errors'      : errors,
                'total'       : len(results),
            }

        except FileNotFoundError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f'Terjadi kesalahan: {e}'

    return render(request, 'detector/index.html', context)
