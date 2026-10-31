# Handoff — Dataset preparation complete

Owner of this phase: **Swapnil Santosh Akhade**

## Status

The BFF-15 dataset has been verified, split, and packaged for training.

Everything you need is importable from `data_utils.py`. Do **not** re-scan,
re-split, or re-augment the dataset inside your model notebook. Import the
helpers instead.

## What is frozen (do not change)

| Item | Value |
|---|---|
| Class list | 15 classes, defined in `CLASS_NAMES` |
| Class → index mapping | `CLASS_TO_IDX`, `IDX_TO_CLASS` |
| Image size | 224 × 224 |
| Batch size | 32 (you can change per model if GPU memory demands it) |
| Random seed | 42 |
| Raw images | 5,577 files in `data/raw/` |
| Byte-identical duplicates removed | 354 (282 duplicate groups) — see note below |
| Unique images used | 5,223 |
| Splits | `data/splits/train.csv` (3,656), `val.csv` (783), `test.csv` (784) |
| Normalization | pixel values scaled to `[0, 1]` |
| Augmentation | horizontal flip, rotation ±18°, zoom ±10%, contrast ±10% — training only |

**Why the counts differ from the published dataset:** BFF-15 ships the same photo
under different filenames, sometimes in different class folders. 81 of the 837
test images in the original split were byte-identical to a train or val image, so
test accuracy was inflated by memorisation. Notebook 00 now hashes every image and
drops duplicates *before* splitting, and section 7 asserts that no two splits share
content. Split sizes therefore changed from 3,903 / 837 / 837.

Split CSVs store paths **relative to the repo root**, so the same files work on
your laptop, on a teammate's machine and in Colab.

## What each model notebook must do

1. Import from `data_utils`:

   ```python
   from data_utils import (
       PROJECT_ROOT,
       get_train_ds, get_val_ds, get_test_ds,
       get_class_weights,
       CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
       IMG_SIZE, BATCH_SIZE, SEED,
   )
   ```

   `PROJECT_ROOT` is derived from `data_utils.__file__`. Do not rebuild it from
   `Path.cwd()` — that writes artifacts to the wrong folder when the notebook is
   opened from a different working directory.

2. Build datasets:

   ```python
   train_ds = get_train_ds()
   val_ds   = get_val_ds()
   test_ds  = get_test_ds()
   ```

3. Use `get_class_weights()` if you decide to weight the loss for the
   imbalanced classes. The weights are already balanced and reproducible.

4. Save every epoch's training/validation metrics to your own log CSV:

   - `custom_cnn/logs/training_log.csv`
   - `transfer_learning/logs/training_log.csv`

   Use `tf.keras.callbacks.CSVLogger`. It writes exactly these columns (the
   metric names come from `compile(..., metrics=["accuracy"])`), plus
   `lr` when you attach `ReduceLROnPlateau`:

   ```
   epoch,accuracy,loss,val_accuracy,val_loss
   ```

   Notebook 03 and `eval_utils.plot_curves()` read those names, so do not
   rename them. Append after every epoch so training can resume if interrupted.

5. Save the best checkpoint to your own folder:

   - `custom_cnn/best_model/`
   - `transfer_learning/best_model/`

   Use `tf.keras.callbacks.ModelCheckpoint(..., save_best_only=True)`, and add
   `eval_utils.ResumeFromCheckpoint()` so a dropped Colab session does not lose
   the run.

6. Save plots to your own folder:

   - `custom_cnn/plots/`
   - `transfer_learning/plots/`

7. Save final test-set metrics to your own folder:

   - `custom_cnn/metrics/final_metrics.json`
   - `transfer_learning/metrics/final_metrics.json`

   Include accuracy, precision, recall, F1 (macro and weighted), and the
   confusion matrix.

   Call the shared helper rather than writing your own — it produces the exact
   schema notebook 03 reads, plus `test_predictions.csv` for the paired
   significance test:

   ```python
   from eval_utils import evaluate_and_save

   evaluate_and_save(best, test_ds, "<model name>", epochs_ran, train_seconds,
                     METRICS_JSON, CM_CSV, PLOTS_DIR / "confusion_matrix.png",
                     trainable_params=trainable_params)
   ```

## Normalization note for the transfer-learning notebook

The shared pipeline produces pixel values in `[0, 1]`.

MobileNetV2 was pretrained on data scaled to `[-1, 1]`. Add one layer at the top
of the model:

```python
import tensorflow as tf
preprocess = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0,
                                       name="pretrained_preprocess")
```

That is precisely what `mobilenet_v2.preprocess_input` computes on `[0, 1]`
input. Use `Rescaling` rather than `Lambda`: a `Lambda` is not serialised safely
into a `.keras` file, so `ModelCheckpoint` + `load_model` needs `safe_mode=False`
and can silently drop the preprocessing. Feeding `[0, 1]` straight into the
backbone is the other wrong option — it works, but degrades the pretrained
features.

Do not duplicate the augmentation. It is already inside `get_train_ds()`.

## Do not

- Do not re-split the dataset in a model notebook; notebook 00 owns the split.
- Do not hand-edit `data/splits/*.csv`. Re-run notebook 00 if they must change.
- Do not modify `data_utils.py` or `eval_utils.py` without telling the group.
- Do not touch the other teammate's model folder.
- Do not commit `best_model/` binaries or the resumable state file (already in
  `.gitignore`). `data/raw/` **is** committed — 40 MB under CC BY 4.0.
