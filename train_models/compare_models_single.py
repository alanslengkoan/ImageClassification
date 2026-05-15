import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

print('✅ TensorFlow version :', tf.__version__)

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR   = '/home/echolog/Documents/Project/www/skripsi/ImageClassification/train_models'
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'output')

MODEL_RESNET      = os.path.join(OUTPUT_DIR, 'resnet50_single_best.h5')
MODEL_MOBILENET   = os.path.join(OUTPUT_DIR, 'mobilenetv2_single_best.keras')
COMPARE_DIR       = os.path.join(OUTPUT_DIR, 'comparison_single')

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

os.makedirs(COMPARE_DIR, exist_ok=True)

# ============================================================
# CEK MODEL TERSEDIA
# ============================================================
for path, name in [(MODEL_RESNET, 'ResNet50 Single'), (MODEL_MOBILENET, 'MobileNetV2 Single')]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Model {name} tidak ditemukan: {path}\n'
            f'Jalankan training terlebih dahulu.'
        )
print('✅ Kedua model ditemukan')

# ============================================================
# LOAD MODEL
# ============================================================
print('\n📦 Loading models...')
model_resnet    = load_model(MODEL_RESNET)
model_mobilenet = load_model(MODEL_MOBILENET)
print('✅ ResNet50 Single loaded')
print('✅ MobileNetV2 Single loaded')

# ============================================================
# DATA GENERATOR — masing-masing pakai preprocessing berbeda
# ============================================================
# ResNet50: preprocess_input ImageNet (caffe-style)
gen_resnet = ImageDataGenerator(preprocessing_function=resnet_preprocess).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# MobileNetV2: preprocess_input ImageNet (tf-style [-1, 1])
gen_mobilenet = ImageDataGenerator(preprocessing_function=mobilenet_preprocess).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

CLASS_NAMES  = list(gen_resnet.class_indices.keys())
CLASS_LABELS = [c.title() for c in CLASS_NAMES]
NUM_CLASSES  = len(CLASS_NAMES)
y_true       = gen_resnet.classes

print(f'\n📊 Test set  : {gen_resnet.samples} gambar')
print(f'   Kelas     : {CLASS_LABELS}')

# ============================================================
# EVALUASI KEDUA MODEL
# ============================================================
print('\n' + '=' * 60)
print('📈 EVALUASI TEST SET')
print('=' * 60)

print('\n🔍 ResNet50 Single...')
gen_resnet.reset()
loss_resnet, acc_resnet = model_resnet.evaluate(gen_resnet, verbose=0)
gen_resnet.reset()
prob_resnet  = model_resnet.predict(gen_resnet, verbose=1)
pred_resnet  = np.argmax(prob_resnet, axis=1)

print('\n🔍 MobileNetV2 Single...')
gen_mobilenet.reset()
loss_mobilenet, acc_mobilenet = model_mobilenet.evaluate(gen_mobilenet, verbose=0)
gen_mobilenet.reset()
prob_mobilenet  = model_mobilenet.predict(gen_mobilenet, verbose=1)
pred_mobilenet  = np.argmax(prob_mobilenet, axis=1)

print(f'\n{"=" * 60}')
print(f'📊 HASIL PERBANDINGAN:')
print(f'{"=" * 60}')
print(f'   {"Model":<25} {"Test Accuracy":>15} {"Test Loss":>12}')
print(f'   {"-"*52}')
print(f'   {"ResNet50 Single":<25} {acc_resnet*100:>14.2f}% {loss_resnet:>12.4f}')
print(f'   {"MobileNetV2 Single":<25} {acc_mobilenet*100:>14.2f}% {loss_mobilenet:>12.4f}')
print(f'{"=" * 60}')
delta = (acc_mobilenet - acc_resnet) * 100
if delta > 0:
    print(f'\n   MobileNetV2 Single unggul {delta:+.2f}% dibanding ResNet50 Single')
elif delta < 0:
    print(f'\n   ResNet50 Single unggul {-delta:+.2f}% dibanding MobileNetV2 Single')
else:
    print(f'\n   Kedua model memiliki accuracy yang sama')

# ============================================================
# 1. BAR CHART: Accuracy & Loss Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Perbandingan ResNet50 vs MobileNetV2 (Single Fine-Tune)\nKlasifikasi Kerusakan Jalan (3 Kelas)',
             fontsize=14, fontweight='bold')

models  = ['ResNet50\nSingle', 'MobileNetV2\nSingle']
colors  = ['#FF9800', '#2196F3']
accs    = [acc_resnet * 100, acc_mobilenet * 100]
losses  = [loss_resnet,      loss_mobilenet]

bars1 = axes[0].bar(models, accs, color=colors, edgecolor='black', width=0.5)
for b, v in zip(bars1, accs):
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].axhline(y=85, color='green', linestyle=':', linewidth=1.5, label='Target 85%')
axes[0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 110)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(models, losses, color=colors, edgecolor='black', width=0.5)
for b, v in zip(bars2, losses):
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[1].set_title('Test Loss', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_single_accuracy_loss.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'\n✅ Grafik accuracy/loss tersimpan: {save_path}')

# ============================================================
# 2. CONFUSION MATRIX SIDE-BY-SIDE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrix — ResNet50 vs MobileNetV2 (Single Fine-Tune)\nKlasifikasi Kerusakan Jalan (3 Kelas)',
             fontsize=14, fontweight='bold')

for ax, pred, title, cmap in [
    (axes[0], pred_resnet,    f'ResNet50 Single (Acc={acc_resnet*100:.2f}%)',      'Oranges'),
    (axes[1], pred_mobilenet, f'MobileNetV2 Single (Acc={acc_mobilenet*100:.2f}%)', 'Blues'),
]:
    cm = confusion_matrix(y_true, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                linewidths=0.5, annot_kws={'size': 13}, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Aktual', fontsize=11)
    ax.set_xlabel('Prediksi', fontsize=11)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_single_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Confusion matrix tersimpan: {save_path}')

# ============================================================
# 3. CLASSIFICATION REPORT — keduanya
# ============================================================
print('\n' + '=' * 60)
print('📋 Classification Report — ResNet50 Single')
print('=' * 60)
report_resnet = classification_report(y_true, pred_resnet, target_names=CLASS_LABELS)
print(report_resnet)

print('=' * 60)
print('📋 Classification Report — MobileNetV2 Single')
print('=' * 60)
report_mobilenet = classification_report(y_true, pred_mobilenet, target_names=CLASS_LABELS)
print(report_mobilenet)

report_path = os.path.join(COMPARE_DIR, 'comparison_single_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('=' * 60 + '\n')
    f.write('PERBANDINGAN MODEL (Single Fine-Tune) — Klasifikasi Kerusakan Jalan (3 Kelas)\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'ResNet50 Single      — Test Accuracy: {acc_resnet*100:.2f}%  | Test Loss: {loss_resnet:.4f}\n')
    f.write(f'MobileNetV2 Single   — Test Accuracy: {acc_mobilenet*100:.2f}%  | Test Loss: {loss_mobilenet:.4f}\n')
    f.write(f'Selisih              — {delta:+.2f}%\n\n')
    f.write('=' * 60 + '\n')
    f.write('Classification Report — ResNet50 Single\n')
    f.write('=' * 60 + '\n')
    f.write(report_resnet)
    f.write('\n' + '=' * 60 + '\n')
    f.write('Classification Report — MobileNetV2 Single\n')
    f.write('=' * 60 + '\n')
    f.write(report_mobilenet)
print(f'✅ Report perbandingan tersimpan: {report_path}')

# ============================================================
# 4. ROC CURVE — keduanya dalam 1 grafik per kelas
# ============================================================
y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
colors_cls = ['#2196F3', '#F44336', '#4CAF50']

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 5))
fig.suptitle('ROC Curve per Kelas — ResNet50 vs MobileNetV2 (Single Fine-Tune)\nKlasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')

for i, (cls, color, ax) in enumerate(zip(CLASS_LABELS, colors_cls, axes)):
    fpr_resnet,    tpr_resnet,    _ = roc_curve(y_bin[:, i], prob_resnet[:, i])
    fpr_mobilenet, tpr_mobilenet, _ = roc_curve(y_bin[:, i], prob_mobilenet[:, i])
    auc_resnet    = auc(fpr_resnet,    tpr_resnet)
    auc_mobilenet = auc(fpr_mobilenet, tpr_mobilenet)

    ax.plot(fpr_resnet,    tpr_resnet,    color='#FF9800', linewidth=2,
            label=f'ResNet50     (AUC={auc_resnet:.2f})')
    ax.plot(fpr_mobilenet, tpr_mobilenet, color='#2196F3', linewidth=2,
            label=f'MobileNetV2  (AUC={auc_mobilenet:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title(f'Kelas: {cls}', fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_single_roc_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ ROC Curve tersimpan: {save_path}')

# ============================================================
# 5. PRECISION-RECALL CURVE — keduanya per kelas
# ============================================================
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 5))
fig.suptitle('Precision-Recall Curve per Kelas — ResNet50 vs MobileNetV2 (Single Fine-Tune)\nKlasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')

for i, (cls, color, ax) in enumerate(zip(CLASS_LABELS, colors_cls, axes)):
    prec_r, rec_r, _ = precision_recall_curve(y_bin[:, i], prob_resnet[:, i])
    prec_m, rec_m, _ = precision_recall_curve(y_bin[:, i], prob_mobilenet[:, i])
    pr_auc_r = auc(rec_r, prec_r)
    pr_auc_m = auc(rec_m, prec_m)

    ax.plot(rec_r, prec_r, color='#FF9800', linewidth=2,
            label=f'ResNet50     (AUC={pr_auc_r:.2f})')
    ax.plot(rec_m, prec_m, color='#2196F3', linewidth=2,
            label=f'MobileNetV2  (AUC={pr_auc_m:.2f})')
    ax.set_title(f'Kelas: {cls}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_single_pr_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Precision-Recall Curve tersimpan: {save_path}')

# ============================================================
# 6. F1-SCORE PER KELAS — grouped bar
# ============================================================
rd_resnet    = classification_report(y_true, pred_resnet,    target_names=CLASS_LABELS, output_dict=True)
rd_mobilenet = classification_report(y_true, pred_mobilenet, target_names=CLASS_LABELS, output_dict=True)

metrics   = ['precision', 'recall', 'f1-score']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Precision / Recall / F1-Score per Kelas\nResNet50 vs MobileNetV2 (Single Fine-Tune)',
             fontsize=13, fontweight='bold')

x     = np.arange(len(CLASS_LABELS))
width = 0.35

for ax, metric in zip(axes, metrics):
    vals_resnet    = [rd_resnet[c][metric]    for c in CLASS_LABELS]
    vals_mobilenet = [rd_mobilenet[c][metric] for c in CLASS_LABELS]

    b1 = ax.bar(x - width/2, vals_resnet,    width, label='ResNet50 Single',     color='#FF9800', edgecolor='black')
    b2 = ax.bar(x + width/2, vals_mobilenet, width, label='MobileNetV2 Single',  color='#2196F3', edgecolor='black')

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_title(metric.capitalize(), fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_single_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik per kelas tersimpan: {save_path}')

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print('\n' + '=' * 60)
print('📋 RINGKASAN PERBANDINGAN MODEL (Single Fine-Tune)')
print('=' * 60)
print(f'   {"":25} {"ResNet50":>14} {"MobileNetV2":>14}')
print(f'   {"-" * 53}')
print(f'   {"Backbone":<25} {"ResNet50":>14} {"MobileNetV2":>14}')
print(f'   {"Strategy":<25} {"Single FT":>14} {"Single FT":>14}')
print(f'   {"Pretrained":<25} {"✅ ImageNet":>14} {"✅ ImageNet":>14}')
print(f'   {"Test Accuracy":<25} {acc_resnet*100:>13.2f}% {acc_mobilenet*100:>13.2f}%')
print(f'   {"Test Loss":<25} {loss_resnet:>14.4f} {loss_mobilenet:>14.4f}')
for cls in CLASS_LABELS:
    f1_r = rd_resnet[cls]['f1-score']
    f1_m = rd_mobilenet[cls]['f1-score']
    print(f'   {f"F1 {cls}":<25} {f1_r:>14.4f} {f1_m:>14.4f}')
print(f'   {"-" * 53}')
if delta > 0:
    print(f'   {"Selisih Accuracy":<25} {f"{delta:+.2f}% (MobileNetV2 unggul)":>28}')
elif delta < 0:
    print(f'   {"Selisih Accuracy":<25} {f"{-delta:+.2f}% (ResNet50 unggul)":>28}')
else:
    print(f'   {"Selisih Accuracy":<25} {"0.00% (Sama)":>28}')
print('=' * 60)
print(f'\n📁 Semua hasil tersimpan di: {COMPARE_DIR}')
