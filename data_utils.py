"""
Shared dataset utilities for BFF-15.

Used by:
- notebooks/00_dataset_setup.ipynb
- notebooks/01_custom_cnn.ipynb
- notebooks/02_transfer_learning.ipynb

Both model notebooks must import from here so that:
- class mapping is identical,
- train/val/test split is identical,
- augmentation pipeline is identical,
- normalization strategy is documented and identical.

This guarantees a fair comparison in notebooks/03_comparison.ipynb.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
AUTOTUNE   = tf.data.AUTOTUNE

# ------------------------------------------------------------------
# Class list (must match folder names inside data/raw/)
# ------------------------------------------------------------------

CLASS_NAMES = [
    "GRASS CARP", "ILISH", "KATLA", "KOI", "MAGUR",
    "MIRROR CARP", "MRIGAL", "PABDA", "PANGAS", "PUTI",
    "RUI", "SHOL", "SILVER CARP", "TELAPIA", "TENGRA",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

# ------------------------------------------------------------------
# Dataset scanning
# ------------------------------------------------------------------

def scan_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Walk data_dir assuming one folder per class.
    Returns a DataFrame with columns: filepath, label, class_name.
    Does not modify anything on disk.
    """
    records = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({
                    "filepath":   str(f),
                    "label":      CLASS_TO_IDX[class_name],
                    "class_name": class_name,
                })
    return pd.DataFrame(records)

# ------------------------------------------------------------------
# Load frozen splits (produced by notebook 00)
# ------------------------------------------------------------------

def load_splits(splits_dir: Path = SPLITS_DIR):
    """
    Load train/val/test CSVs saved by notebook 00.
    Returns (train_df, val_df, test_df).
    """
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")
    return train_df, val_df, test_df

# ------------------------------------------------------------------
# Augmentation (training set only)
# ------------------------------------------------------------------

def get_augmentation() -> tf.keras.Sequential:
    """
    Mild on-the-fly augmentation.
    Applied ONLY to the training set.
    Disabled at inference time.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),   # ±18°
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")

# ------------------------------------------------------------------
# Per-image pipeline
# ------------------------------------------------------------------

def _load_image(path, label):
    """
    Read + decode + resize + normalize to [0, 1].
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# ------------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------------

def make_dataset(df_split, batch_size=BATCH_SIZE, augment=False, shuffle=False):
    """
    Build a tf.data.Dataset from a split DataFrame.

    - augment: apply augmentation map (training only)
    - shuffle: shuffle before mapping (training only)
    - output pixel range is guaranteed in [0, 1]
    """
    paths  = df_split["filepath"].values
    labels = df_split["label"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df_split),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)

    if augment:
        aug = get_augmentation()
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(x, 0.0, 1.0), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# ------------------------------------------------------------------
# Convenience wrappers for teammates
# ------------------------------------------------------------------

def get_train_ds(batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    train_df, _, _ = load_splits()
    return make_dataset(train_df, batch_size=batch_size, augment=True, shuffle=True)

def get_val_ds(batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    _, val_df, _ = load_splits()
    return make_dataset(val_df, batch_size=batch_size, augment=False, shuffle=False)

def get_test_ds(batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    _, _, test_df = load_splits()
    return make_dataset(test_df, batch_size=batch_size, augment=False, shuffle=False)

# ------------------------------------------------------------------
# Class weights (optional, for imbalance)
# ------------------------------------------------------------------

def get_class_weights():
    """
    Compute balanced class weights from the training split.
    Returns a dict {class_index: weight}.
    """
    train_df, _, _ = load_splits()
    counts = train_df["label"].value_counts().sort_index()

    n_total   = int(counts.sum())
    n_classes = len(CLASS_NAMES)

    weights = {}
    for label, count in zip(counts.index.to_numpy(), counts.to_numpy()):
        weights[int(label)] = float(n_total / (n_classes * int(count)))
    return weights