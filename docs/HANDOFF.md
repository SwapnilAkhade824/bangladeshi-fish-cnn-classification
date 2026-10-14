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
| Splits | `data/splits/train.csv` (3,903), `val.csv` (837), `test.csv` (837) |
| Normalization | pixel values scaled to `[0, 1]` |
| Augmentation | horizontal flip, rotation ±18°, zoom ±10%, contrast ±10% — training only |

## What each model notebook must do

1. Import from `data_utils`:

   ```python
   from data_utils import (
       get_train_ds, get_val_ds, get_test_ds,
       get_class_weights,
       CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
       IMG_SIZE, BATCH_SIZE, SEED,
   )
   ```

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

   Suggested columns:

   ```
   epoch, train_loss, train_accuracy, val_loss, val_accuracy, lr, timestamp
   ```

   Append after every epoch so training can resume if interrupted.

5. Save the best checkpoint to your own folder:

   - `custom_cnn/best_model/`
   - `transfer_learning/best_model/`

   Use `tf.keras.callbacks.ModelCheckpoint(..., save_best_only=True)`.

6. Save plots to your own folder:

   - `custom_cnn/plots/`
   - `transfer_learning/plots/`

7. Save final test-set metrics to your own folder:

   - `custom_cnn/metrics/final_metrics.json`
   - `transfer_learning/metrics/final_metrics.json`

   Include accuracy, precision, recall, F1 (macro and weighted), and the
   confusion matrix.

## Normalization note for the transfer-learning notebook

The shared pipeline produces pixel values in `[0, 1]`.

Most pretrained models (MobileNetV2, DenseNet121, etc.) expect their own
preprocessing. Add one layer at the top of the model:

```python
import tensorflow as tf
preprocess = tf.keras.layers.Lambda(
    lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0),
    name="pretrained_preprocess",
)
```

That way both models consume the same `[0, 1]` input, and the pretrained
model gets exactly what it was trained on.

Do not duplicate the augmentation. It is already inside `get_train_ds()`.

## Do not

- Do not re-split the dataset.
- Do not regenerate `data/splits/*.csv`.
- Do not modify `data_utils.py` without telling the group.
- Do not touch the other teammate's model folder.
- Do not commit `data/raw/` or `best_model/` binaries to git.
