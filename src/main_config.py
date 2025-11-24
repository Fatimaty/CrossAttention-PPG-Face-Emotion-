
import os, sys, random, re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

np.set_printoptions(linewidth=120)
print('Python:', sys.version)

print('TensorFlow:', tf.__version__)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detected:', gpus)
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print('Memory growth set error:', e)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Root folder: subject -> emotion -> {CWT, Facial_frames}
base_dir = r"E:\Update_data\processed_data"

# Input shapes
CWT_SHAPE  = (128, 128, 1)  # grayscale
FACE_SHAPE = (64,  64,  3)  # RGB

EMOTIONS = ["happy", "sad", "calm", "angry"]
EMO2IDX = {e:i for i, e in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)

# Pairing settings (from your loader)
KEY_MODE   = "number"          # "number" or "stem"
KEY_REGEX  = r"(\d{1,6})"      # numeric pattern to capture indices
PAD_WIDTH  = 3                 # zero padding for keys
OFFSET     = 0                 # shift (useful if face index = cwt index + offset)

# Training
EPOCHS = 50
BATCH_SIZE = 8
USE_CLASS_WEIGHTS = True

# Optimizer & callbacks
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4

# Output directory (best-only checkpoints go here)
WEIGHT_DIR = "weights_loso_pairkey"
os.makedirs(WEIGHT_DIR, exist_ok=True)

print('Base dir:', base_dir)
print('Emotions:', EMOTIONS)
print('Shapes -> CWT:', CWT_SHAPE, ' Face:', FACE_SHAPE)
print('Pairing -> key_mode:', KEY_MODE, ' key_regex:', KEY_REGEX, ' pad_width:', PAD_WIDTH, ' offset:', OFFSET)
print('Epochs:', EPOCHS, ' Batch size:', BATCH_SIZE, ' Class weights:', USE_CLASS_WEIGHTS)

