import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR    = '/home/echolog/Documents/Project/www/skripsi/ImageClassification-CNN'
MODEL_PATH  = os.path.join(BASE_DIR, 'dataset', 'output', 'resnet50_3class_best.h5')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'dataset', 'output')

IMG_SIZE    = (224, 224)
CLASS_NAMES = ['baik', 'berat', 'sedang']   # urutan alphabetical dari ImageDataGenerator
CLASS_LABELS = {
    'baik'  : ('Baik',   '#4CAF50', '🟢'),   # hijau
    'sedang': ('Sedang', '#FF9800', '🟡'),   # oranye
    'berat' : ('Berat',  '#F44336', '🔴'),   # merah
}

# ============================================================
# LOAD MODEL
# ============================================================
def load_trained_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model tidak ditemukan: {model_path}\n'
            f'Pastikan sudah menjalankan resnet50.py terlebih dahulu.'
        )
    print(f'✅ Memuat model dari: {model_path}')
    model = load_model(model_path)
    print(f'   Input shape  : {model.input_shape}')
    print(f'   Output shape : {model.output_shape}')
    return model

# ============================================================
# PREPROCESS GAMBAR
# ============================================================
def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

# ============================================================
# PREDIKSI SATU GAMBAR
# ============================================================
def predict_image(model, image_path: str, show_plot: bool = True) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Gambar tidak ditemukan: {image_path}')

    img_input = preprocess_image(image_path)
    probs     = model.predict(img_input, verbose=0)[0]

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    label, color, icon = CLASS_LABELS[pred_class]

    result = {
        'image_path' : image_path,
        'class'      : pred_class,
        'label'      : label,
        'confidence' : confidence,
        'probabilities': {CLASS_NAMES[i]: float(probs[i]) * 100 for i in range(len(CLASS_NAMES))},
    }

    print(f'\n{"=" * 55}')
    print(f'🔍 Hasil Deteksi Kerusakan Jalan')
    print(f'{"=" * 55}')
    print(f'   File        : {os.path.basename(image_path)}')
    print(f'   Prediksi    : {icon} {label}')
    print(f'   Confidence  : {confidence:.2f}%')
    print(f'\n   Probabilitas per kelas:')
    for cls in CLASS_NAMES:
        lbl, _, ic = CLASS_LABELS[cls]
        bar = '█' * int(result["probabilities"][cls] / 5)
        print(f'   {ic} {lbl:8s}: {result["probabilities"][cls]:6.2f}%  {bar}')
    print(f'{"=" * 55}')

    if show_plot:
        _plot_result(image_path, result, color, label, icon)

    return result

# ============================================================
# PREDIKSI BATCH (FOLDER)
# ============================================================
def predict_folder(model, folder_path: str, save_summary: bool = True) -> list:
    exts   = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]

    if not images:
        print(f'⚠️  Tidak ada gambar di folder: {folder_path}')
        return []

    print(f'\n📂 Memproses {len(images)} gambar dari: {folder_path}')
    print('-' * 55)

    results = []
    counts  = {cls: 0 for cls in CLASS_NAMES}

    for fname in sorted(images):
        fpath  = os.path.join(folder_path, fname)
        result = predict_image(model, fpath, show_plot=False)
        results.append(result)
        counts[result['class']] += 1

    print(f'\n📊 Ringkasan Prediksi Batch ({len(images)} gambar):')
    print('-' * 45)
    for cls in CLASS_NAMES:
        lbl, _, ic = CLASS_LABELS[cls]
        pct = counts[cls] / len(images) * 100
        print(f'   {ic} {lbl:8s}: {counts[cls]:3d} gambar ({pct:.1f}%)')
    print('-' * 45)

    if save_summary:
        _plot_batch_summary(results, counts, folder_path)

    return results

# ============================================================
# PLOT HASIL SATU GAMBAR
# ============================================================
def _plot_result(image_path: str, result: dict, color: str, label: str, icon: str):
    img = Image.open(image_path).convert('RGB')

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Deteksi Kerusakan Jalan — ResNet50 Fine-Tuning (3 Kelas)',
                 fontsize=13, fontweight='bold')

    ax_img.imshow(img)
    ax_img.set_title(f'{icon}  Prediksi: {label}  ({result["confidence"]:.2f}%)',
                     fontsize=12, color=color, fontweight='bold')
    ax_img.axis('off')

    classes = CLASS_NAMES
    probs   = [result['probabilities'][c] for c in classes]
    colors  = [CLASS_LABELS[c][1] for c in classes]
    bars    = ax_bar.barh(classes, probs, color=colors, edgecolor='black', height=0.5)

    for bar, p in zip(bars, probs):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{p:.2f}%', va='center', fontsize=10)

    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel('Probabilitas (%)', fontsize=11)
    ax_bar.set_title('Distribusi Probabilitas', fontsize=11)
    ax_bar.axvline(x=50, color='gray', linestyle=':', linewidth=1)
    ax_bar.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    fname     = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f'predict_{fname}.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅ Hasil disimpan: {save_path}')

# ============================================================
# PLOT RINGKASAN BATCH
# ============================================================
def _plot_batch_summary(results: list, counts: dict, folder_path: str):
    labels  = [CLASS_LABELS[c][0] for c in CLASS_NAMES]
    sizes   = [counts[c] for c in CLASS_NAMES]
    colors  = [CLASS_LABELS[c][1] for c in CLASS_NAMES]
    explode = [0.05] * len(CLASS_NAMES)

    fig, (ax_pie, ax_conf) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Ringkasan Prediksi Batch — {len(results)} Gambar\nFolder: {os.path.basename(folder_path)}',
                 fontsize=12, fontweight='bold')

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        explode=explode, startangle=90,
        textprops={'fontsize': 11}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax_pie.set_title('Distribusi Kelas Prediksi')

    confs = [r['confidence'] for r in results]
    clrs  = [CLASS_LABELS[r['class']][1] for r in results]
    ax_conf.scatter(range(len(confs)), confs, c=clrs, s=60, edgecolors='black', linewidths=0.5)
    ax_conf.axhline(y=50, color='gray', linestyle=':', linewidth=1)
    ax_conf.set_xlabel('Index Gambar'); ax_conf.set_ylabel('Confidence (%)')
    ax_conf.set_title('Confidence per Gambar')
    ax_conf.set_ylim(0, 105); ax_conf.grid(True, alpha=0.3)

    patches = [mpatches.Patch(color=CLASS_LABELS[c][1], label=CLASS_LABELS[c][0]) for c in CLASS_NAMES]
    ax_conf.legend(handles=patches, loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'predict_batch_summary.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅ Ringkasan batch disimpan: {save_path}')

# ============================================================
# MAIN — jalankan langsung via CLI
# ============================================================
if __name__ == '__main__':
    model = load_trained_model()

    if len(sys.argv) < 2:
        print('\n📖 Cara penggunaan:')
        print('   Satu gambar : python3 predict.py /path/ke/gambar.jpg')
        print('   Satu folder : python3 predict.py /path/ke/folder/')
        sys.exit(0)

    target = sys.argv[1]

    if os.path.isdir(target):
        predict_folder(model, target)
    elif os.path.isfile(target):
        predict_image(model, target)
    else:
        print(f'❌ Path tidak ditemukan: {target}')
        sys.exit(1)
