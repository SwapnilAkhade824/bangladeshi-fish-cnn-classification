"""
Shared dataset utilities for BFF-15.

Used by:
- notebooks/00_dataset_setup.ipynb
- notebooks/01_custom_cnn.ipynb
- notebooks/02_transfer_learning.ipynb
- eval_utils.py (shared by all three model/comparison notebooks)

Both model notebooks must import from here so that:
- class mapping is identical,
- train/val/test split is identical,
- augmentation pipeline is identical,
- normalization strategy is documented and identical.

This guarantees a fair comparison in notebooks/03_comparison.ipynb.

Split membership is owned by notebook 00 and frozen in data/splits/*.csv.
Notebook 00 de-duplicates byte-identical images before splitting; never
re-split inside a model notebook.
"""

from pathlib import Path
import hashlib
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
# Portable paths
# ------------------------------------------------------------------

def to_rel(path) -> str:
    """
    Store a dataset path relative to PROJECT_ROOT (posix separators).

    Split CSVs are committed to git and opened on several machines
    (local Windows checkout, Google Colab, teammate laptops). An
    absolute path recorded on one machine resolves nowhere else, so
    only relative paths are ever written to disk.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # not inside the project (e.g. Colab mounts data elsewhere)
        return str(p)

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
                    "filepath":   to_rel(f),
                    "label":      CLASS_TO_IDX[class_name],
                    "class_name": class_name,
                })
    return pd.DataFrame(records)

# ------------------------------------------------------------------
# Content de-duplication (must run BEFORE splitting)
# ------------------------------------------------------------------

def _anchor(path) -> Path:
    """Absolute paths pass through; relative paths are taken from PROJECT_ROOT."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

def content_hash(path) -> str:
    """MD5 of the raw bytes, streamed so large files do not load into RAM.

    Relative paths are anchored to PROJECT_ROOT, not to the caller's cwd, so a
    notebook opened from `notebooks/` hashes the same bytes as one opened from
    the repo root.
    """
    h = hashlib.md5()
    with open(_anchor(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def content_dedupe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drop byte-identical images, keeping one representative per content hash.

    BFF-15 ships the same photo under different filenames (and sometimes in
    a different class folder). A plain path-based split therefore puts one
    copy in train and another in test, which inflates test accuracy. Hashing
    the bytes first makes the split leakage-proof.

    The file with the lowest name is kept so the result is deterministic and
    independent of the OS directory listing order.
    """
    frame = df.copy()
    frame["_hash"] = [content_hash(p) for p in frame["filepath"]]
    frame = frame.sort_values("filepath", key=lambda s: s.str.lower())
    kept = frame.drop_duplicates(subset="_hash", keep="first")
    kept = kept.sort_index().drop(columns="_hash").reset_index(drop=True)

    if verbose:
        removed = len(frame) - len(kept)
        n_groups = int(pd.Series(frame["_hash"].value_counts() > 1).sum())
        print(f"content-dedupe: {len(df)} images -> {len(kept)} "
              f"({removed} byte-identical copies removed across {n_groups} duplicate groups)")
    return kept

# ------------------------------------------------------------------
# Load frozen splits (produced by notebook 00)
# ------------------------------------------------------------------

def load_splits(splits_dir: Path = SPLITS_DIR):
    """
    Load train/val/test CSVs saved by notebook 00.
    Returns (train_df, val_df, test_df) with `filepath` resolved to an
    absolute path for the machine doing the loading.

    Older copies of these CSVs stored absolute paths from the machine
    that generated them; such rows are rewritten from `class_name` +
    filename so a previously committed split still loads here.
    """
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df["filepath"] = df["filepath"].map(_resolve_path)
        df["label"]    = df["class_name"].map(CLASS_TO_IDX)  # never trust a stale index
        present = df["filepath"].map(lambda p: _anchor(p).exists())
        missing = df[~present]
        if len(missing):
            raise FileNotFoundError(
                f"{name}.csv: {len(missing)} image(s) not found, e.g. "
                f"{missing['filepath'].iloc[0]}. Check that data/raw/ is "
                f"present, or re-run notebooks/00_dataset_setup.ipynb."
            )
    return train_df, val_df, test_df

def _resolve_path(value: str) -> str:
    """Relative -> absolute; legacy absolute -> rebuilt from tail components."""
    p = Path(value)
    if not p.is_absolute():
        return str(PROJECT_ROOT / p)
    if p.exists():
        return str(p)
    # e.g. D:\Dev\College\...\data\raw\RUI\RUI_0147.jpg -> data/raw/RUI/RUI_0147.jpg
    parts = p.parts
    if "raw" in parts:
        tail = parts[parts.index("raw") + 1:]
        candidate = DATA_DIR.joinpath(*tail)
        if candidate.exists():
            return str(candidate)
    return str(p)

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