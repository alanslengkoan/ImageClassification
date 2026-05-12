from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-skripsi-road-damage-detector-secret-key-2024'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'road_detector.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'road_detector.wsgi.application'

LANGUAGE_CODE = 'id'
TIME_ZONE     = 'Asia/Makassar'
USE_I18N      = True
USE_TZ        = True

STATIC_URL   = '/static/'
MEDIA_URL    = '/media/'
MEDIA_ROOT   = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Path model terlatih
MODEL_PATH = (
    BASE_DIR.parent / 'models' / 'resnet50_3class_best.h5'
)

# Batas upload: 100 MB total (untuk multiple files)
DATA_UPLOAD_MAX_MEMORY_SIZE  = 100 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE  = 10  * 1024 * 1024   # maks 10 MB per file
MAX_UPLOAD_FILES             = 10                   # maks 10 file sekaligus
