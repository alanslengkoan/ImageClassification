# ============================================================
# BASELINE CNN — DILATIH DARI NOL (TANPA TRANSFER LEARNING)
# Fungsi: pembanding dasar untuk ResNet50 & MobileNetV2
# ============================================================

import os
import random

# ============================================================
# REPRODUCIBILITY — harus sebelum import TensorFlow
# ============================================================
SEED = 42
os.environ['PYTHONHASHSEED']        = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from sklearn.preprocessing import label_binarize
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

# Reproducibility — set semua seed
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print('✅ TensorFlow version :', tf.__version__)
print('✅ GPU tersedia       :', tf.config.list_physical_devices('GPU'))

# ============================================================
# PATH
# ============================================================
BASE_DIR   = '/home/echolog/Documents/Project/www/skripsi/ImageClassification/train_models'

TRAIN_DIR  = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR    = os.path.join(BASE_DIR, 'dataset', 'val')
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')

OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
NUM_CLASSES   = 3

EPOCHS        = 80
LEARNING_RATE = 0.001

print('\n============================================================')
print('⚙️ KONFIGURASI TRAINING — BASELINE CNN')
print('============================================================')

print(f'IMG_SIZE            : {IMG_SIZE}')
print(f'BATCH_SIZE          : {BATCH_SIZE}')
print(f'Epochs              : {EPOCHS} (LR={LEARNING_RATE})')
print(f'Arsitektur          : CNN from scratch (tanpa pretrained)')

# ============================================================
# AUGMENTASI — rescale 1/255 (tanpa preprocess_input pretrained)
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,

    shear_range=0.10,
    zoom_range=0.15,

    horizontal_flip=True,
    vertical_flip=False,

    brightness_range=[0.85, 1.15],

    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# TTA augmentation (lighter)
tta_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    horizontal_flip=True,
    zoom_range=0.08,
    fill_mode='nearest'
)

# ============================================================
# GENERATOR
# ============================================================
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

CLASS_NAMES  = list(train_generator.class_indices.keys())
CLASS_LABELS = [c.title() for c in CLASS_NAMES]

print('\n============================================================')
print('📊 DATASET')
print('============================================================')

print(f'Train : {train_generator.samples}')
print(f'Val   : {val_generator.samples}')
print(f'Test  : {test_generator.samples}')
print(f'Class : {CLASS_LABELS}')

# ============================================================
# MODEL — BASELINE CNN (4 blok konvolusi)
# ============================================================
inputs = keras.Input(shape=(224, 224, 3))

# Block 1
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 2
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 3
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 4
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Head
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(0.001)
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation='softmax'
)(x)

model = keras.Model(inputs, outputs)

model.summary()

# ============================================================
# COMPILE
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.10
    ),

    metrics=['accuracy']
)

print('\n✅ Model berhasil dikompilasi (Baseline CNN)')

# ============================================================
# CALLBACKS
# ============================================================
model_save_path = os.path.join(
    OUTPUT_DIR,
    'baseline_cnn_best.keras'
)

callbacks = [

    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# TRAINING
# ============================================================
print('\n============================================================')
print('🚀 TRAINING — BASELINE CNN')
print('============================================================')

history_obj = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

history = history_obj.history

print('\n✅ Training selesai')
print(f'Best Val Accuracy: {max(history["val_accuracy"])*100:.2f}%')

# ============================================================
# HASIL TRAINING
# ============================================================
best_val_acc = max(history['val_accuracy'])

print('\n============================================================')
print('📊 HASIL TRAINING')
print('============================================================')

print(f'Best Val Accuracy : {best_val_acc*100:.2f}%')
print(f'Total Epochs Run  : {len(history["accuracy"])}')

# ============================================================
# VISUALISASI
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['accuracy'], label='Train')
ax1.plot(history['val_accuracy'], label='Validation')

ax1.axhline(y=0.85, color='green', linestyle='--', label='Target 85%')

ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history['loss'], label='Train')
ax2.plot(history['val_loss'], label='Validation')

ax2.set_title('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    'baseline_cnn_history.png'
)

plt.savefig(save_path, dpi=150)
plt.show()

print(f'✅ Grafik tersimpan: {save_path}')

# ============================================================
# EVALUASI TEST SET
# ============================================================
print('\n============================================================')
print('🔍 EVALUASI TEST SET')
print('============================================================')

test_generator.reset()
test_loss, test_acc = model.evaluate(
    test_generator,
    verbose=1
)

print(f'\n🏆 Test Accuracy (standard) : {test_acc*100:.2f}%')
print(f'📉 Test Loss                : {test_loss:.4f}')

# ============================================================
# TTA (TEST TIME AUGMENTATION) — 5 PASSES
# ============================================================
print('\n🔄 Menjalankan TTA (5 augmented passes)...')

NUM_TTA = 5
y_true = test_generator.classes
n_samples = len(y_true)

# Standard prediction
test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=0)

# TTA predictions
for tta_i in range(NUM_TTA):
    tta_gen = tta_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    tta_pred = model.predict(tta_gen, verbose=0)
    y_pred_prob += tta_pred

# Average over (1 standard + NUM_TTA augmented)
y_pred_prob = y_pred_prob / (1 + NUM_TTA)

y_pred = np.argmax(y_pred_prob, axis=1)

tta_acc = np.mean(y_pred == y_true)
print(f'🏆 Test Accuracy (TTA)      : {tta_acc*100:.2f}%')

# Use TTA accuracy as final
test_acc = tta_acc

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=CLASS_LABELS,
    yticklabels=CLASS_LABELS
)

plt.title('Confusion Matrix — Baseline CNN')

plt.xlabel('Prediksi')
plt.ylabel('Aktual')

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    'baseline_cnn_confusion_matrix.png'
)

plt.savefig(save_path, dpi=150)
plt.show()

print(f'✅ Confusion Matrix tersimpan: {save_path}')

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print('\n============================================================')
print('📋 CLASSIFICATION REPORT')
print('============================================================')

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_LABELS
)

print(report)

report_path = os.path.join(OUTPUT_DIR, 'baseline_cnn_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('Classification Report — Baseline CNN (from scratch) + TTA\n')
    f.write('Task      : Klasifikasi Kerusakan Jalan\n')
    f.write('Kelas     : Baik | Sedang | Berat\n')
    f.write('='*60 + '\n')
    f.write(report)
    f.write(f'\nTest Accuracy (TTA) : {test_acc*100:.2f}%')
    f.write(f'\nTest Loss           : {test_loss:.4f}')
print(f'✅ Report tersimpan: {report_path}')

# ============================================================
# ROC CURVE
# ============================================================
y_bin = label_binarize(
    y_true,
    classes=list(range(NUM_CLASSES))
)

plt.figure(figsize=(9, 6))

colors = cycle(['blue', 'red', 'green'])

for i, (color, cls) in enumerate(zip(colors, CLASS_LABELS)):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        y_pred_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        color=color,
        linewidth=2,
        label=f'{cls} (AUC = {roc_auc:.2f})'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.title('ROC Curve — Baseline CNN')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.grid(True)

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    'baseline_cnn_roc_curve.png'
)

plt.savefig(save_path, dpi=150)
plt.show()

print(f'✅ ROC Curve tersimpan: {save_path}')

# ============================================================
# PRECISION-RECALL CURVE
# ============================================================
plt.figure(figsize=(9, 6))

colors = cycle(['blue', 'red', 'green'])

for i, (color, cls) in enumerate(zip(colors, CLASS_LABELS)):

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_bin[:, i],
        y_pred_prob[:, i]
    )

    pr_auc = auc(recall_vals, precision_vals)

    plt.plot(
        recall_vals,
        precision_vals,
        color=color,
        linewidth=2,
        label=f'{cls} (AUC = {pr_auc:.2f})'
    )

plt.title('Precision-Recall Curve — Baseline CNN + TTA')

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend()
plt.grid(True)

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    'baseline_cnn_pr_curve.png'
)

plt.savefig(save_path, dpi=150)
plt.show()

print(f'✅ Precision-Recall Curve tersimpan: {save_path}')

# ============================================================
# GRAFIK PRECISION, RECALL & F1 PER KELAS
# ============================================================
report_dict    = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
precision_list = [report_dict[c]['precision'] for c in CLASS_LABELS]
recall_list    = [report_dict[c]['recall']    for c in CLASS_LABELS]
f1_list        = [report_dict[c]['f1-score']  for c in CLASS_LABELS]

x_pos, width = np.arange(len(CLASS_LABELS)), 0.25
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x_pos - width, precision_list, width, label='Precision', color='#2196F3', edgecolor='black')
b2 = ax.bar(x_pos,         recall_list,    width, label='Recall',    color='#4CAF50', edgecolor='black')
b3 = ax.bar(x_pos + width, f1_list,        width, label='F1-Score',  color='#FF5722', edgecolor='black')
for bars in [b1, b2, b3]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Kelas Kerusakan Jalan', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall & F1-Score per Kelas\nBaseline CNN + TTA — Klasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(CLASS_LABELS)
ax.set_ylim(0, 1.2); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'baseline_cnn_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik per kelas tersimpan: {save_path}')

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print('\n============================================================')
print('📋 RINGKASAN AKHIR')
print('============================================================')

print(f'Arsitektur            : Baseline CNN (from scratch)')
print(f'Best Val Accuracy     : {best_val_acc*100:.2f}%')
print(f'Test Accuracy         : {test_acc*100:.2f}%')
print(f'Test Loss             : {test_loss:.4f}')

print(f'Conv Blocks           : 4 (32→64→128→128)')
print(f'Dropout               : 0.50')
print(f'Label Smoothing       : 0.10')
print(f'Strategy              : From Scratch + TTA')

print(f'Model Saved           : baseline_cnn_best.keras')

print('============================================================')

if test_acc >= 0.85:
    print('\n🔥 TARGET TEST ACCURACY ≥ 85% BERHASIL DICAPAI!')
elif test_acc >= 0.80:
    print('\n✅ TEST ACCURACY > 80%, MENDEKATI TARGET 85%')
else:
    print(f'\n⚠️ TEST ACCURACY MASIH : {test_acc*100:.2f}% (wajar untuk baseline tanpa transfer learning)')
