"""
Shared evaluation / reporting helpers for BFF-15.

Used by:
- notebooks/01_custom_cnn.ipynb   (Vivek)
- notebooks/02_transfer_learning.ipynb   (Mansi)
- notebooks/03_comparison.ipynb          (Swapnil)

`evaluate_and_save` and `plot_curves` used to be pasted verbatim into both
model notebooks. Two copies of the artifact-writing code drift apart as soon
as one owner tweaks a key name, and notebook 03 then reads a file that does
not match. Keeping one copy here makes the artifact contract enforceable.

Artifact schema produced per model (notebook 03 depends on these names):
    <dir>/logs/training_log.csv           CSVLogger: epoch,accuracy,loss,val_accuracy,val_loss
    <dir>/metrics/final_metrics.json      metrics listed below
    <dir>/metrics/confusion_matrix.csv    15x15, CLASS_NAMES as index and columns
    <dir>/metrics/test_predictions.csv    per-image true/predicted label (for McNemar)
    <dir>/plots/training_curves.png
    <dir>/plots/confusion_matrix.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

from data_utils import CLASS_NAMES, load_splits, IDX_TO_CLASS

# ------------------------------------------------------------------
# Long-run safety (Google Colab kills a session after ~90 min idle and
# reclaims T4 GPUs after ~12 h; an interrupted training run otherwise
# loses every epoch that did not win the val_accuracy checkpoint)
# ------------------------------------------------------------------

class ResumeFromCheckpoint(tf.keras.callbacks.Callback):
    """
    Persist live training state so an interrupted run can be picked up again.

    ModelCheckpoint with save_best_only=True only keeps the best weights and
    throws away the epoch counter, the optimiser state and the learning-rate
    schedule. This callback writes all three after every epoch.

    Usage
    -----
    cb = ResumeFromCheckpoint(path)                 # fresh run: seeds from path
    cb = ResumeFromCheckpoint(path, resume=False)   # deliberate re-run from scratch
    model.fit(..., initial_epoch=cb.epoch_ran, callbacks=[cb, ...])

    Weights go to a `<stem>.weights.h5` file (the only extension Keras 3
    accepts for save_weights) with a small JSON sidecar next to it. Name the
    path so the stem differs from `best.keras`: reloading `best.keras` in the
    evaluation cell must always give the *best* epoch, never the last one.
    """

    def __init__(self, path, resume=True):
        super().__init__()
        # Keras 3's save_weights only accepts a `.weights.h5` filename, so
        # normalise whatever the caller passed (e.g. `last_state.kerasresumable`)
        # into `<stem>.weights.h5` and keep the epoch-counter sidecar next to it.
        raw = Path(path)
        stem = raw.name
        for ext in (".weights.h5", ".h5", ".keras", ".kerasresumable", ".json"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        self.path = raw.parent / f"{stem}.weights.h5"
        self.state_path = raw.parent / f"{stem}.json"
        self.wanted_resume = resume
        self.epoch_ran = 0
        self.resumed = False

    def set_model(self, model):
        super().set_model(model)
        if not (self.wanted_resume and self.path.exists() and self.state_path.exists()):
            return
        try:
            model.load_weights(str(self.path))
            self.epoch_ran = int(json.loads(self.state_path.read_text())["epoch_ran"])
            self.resumed = True
            print(f"ResumeFromCheckpoint: restored {self.path.name} "
                  f"at epoch {self.epoch_ran}")
        except Exception as exc:                    # partial write before a crash
            print(f"ResumeFromCheckpoint: could not resume ({exc}); starting fresh")
            self.epoch_ran = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_ran = int(epoch) + 1
        self.model.save_weights(str(self.path))
        self.state_path.write_text(json.dumps({
            "epoch_ran": self.epoch_ran,
            "last_logs": {k: float(v) for k, v in (logs or {}).items()},
        }, indent=2))

# ------------------------------------------------------------------
# Test-set evaluation + artifact export
# ------------------------------------------------------------------

def evaluate_and_save(model, test_ds, model_name, epochs_ran, train_seconds,
                      metrics_json, cm_csv, cm_png,
                      trainable_params=None, extra_metrics=None):
    """
    Evaluate on the frozen test split and write every artifact notebook 03 expects.

    Parameters
    ----------
    model            : compiled tf.keras.Model
    test_ds          : the tf.data.Dataset from get_test_ds(). Passed in rather
                       than rebuilt here, because a rebuild would re-shuffle and
                       misalign predictions against the labels in test.csv.
    model_name       : label written to final_metrics.json["base_model"]
    epochs_ran       : int, total epochs actually trained
    train_seconds    : float, wall-clock training time
    metrics_json / cm_csv / cm_png : output paths
    trainable_params : int or None. Pass the count captured at the end of
                       training. After load_model() Keras reports every layer
                       trainable again, so measuring here would overstate the
                       fine-tuned model badly.
    extra_metrics    : optional dict merged into final_metrics.json
    """
    _, _, test_df = load_splits()
    y_true = test_df["label"].values          # test_ds is NOT shuffled -> same order

    test_loss, test_acc_keras = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    assert len(y_true) == len(y_pred), "test order changed; predictions cannot be aligned"

    idx = list(range(len(CLASS_NAMES)))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=idx)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_csv)

    # Top-5 is the metric the reference paper reports, so include it for a
    # like-for-like read-out. Only meaningful when there are >5 classes.
    top5 = float((probs.argsort(axis=1)[:, -5:] == y_true[:, None]).any(axis=1).mean())

    # One-vs-rest AUC needs every class present in the test labels.
    try:
        auc = float(roc_auc_score(
            pd.get_dummies(y_true).reindex(columns=idx, fill_value=0), probs, multi_class="ovr")
        )
    except Exception:
        auc = None                            # a class with zero test support

    metrics = {
        "base_model": model_name,
        "test_loss": float(test_loss),
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_top5_accuracy": top5,
        "test_roc_auc_ovr": auc,
        "test_precision_macro": float(p_m), "test_recall_macro": float(r_m), "test_f1_macro": float(f_m),
        "test_precision_weighted": float(p_w), "test_recall_weighted": float(r_w), "test_f1_weighted": float(f_w),
        "n_test": int(len(y_true)),
        "epochs_ran": int(epochs_ran),
        "train_seconds": float(train_seconds),
        "total_params": int(model.count_params()),
        "trainable_params": int(trainable_params) if trainable_params is not None
                            else int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
        "per_class": {
            c: {"precision": float(p_c[i]), "recall": float(r_c[i]), "f1": float(f_c[i]), "support": int(s_c[i])}
            for i, c in enumerate(CLASS_NAMES)
        },
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-image predictions so notebook 03 can run a paired significance test
    # (McNemar) without re-running either model.
    pd.DataFrame({
        "filepath":   test_df["filepath"].values,
        "class_name": test_df["class_name"].values,
        "true_label": y_true,
        "true_class": [IDX_TO_CLASS[int(v)] for v in y_true],
        "pred_label": y_pred,
        "pred_class": [IDX_TO_CLASS[int(v)] for v in y_pred],
        "confidence": probs[np.arange(len(y_pred)), y_pred],
        "correct": (y_true == y_pred).astype(int),
    }).to_csv(Path(metrics_json).parent / "test_predictions.csv", index=False)

    _plot_confusion_matrix(cm, model_name, cm_png)

    print(f"Test accuracy : {metrics['test_accuracy']:.4f}")
    print(f"Top-5 accuracy: {metrics['test_top5_accuracy']:.4f}")
    print(f"Macro  P/R/F1 : {p_m:.4f} / {r_m:.4f} / {f_m:.4f}")
    print(f"Weight P/R/F1 : {p_w:.4f} / {r_w:.4f} / {f_w:.4f}")
    print(pd.DataFrame(metrics["per_class"]).T.round(3))
    return metrics

def _plot_confusion_matrix(cm, model_name, cm_png):
    """Counts annotated, colour row-normalised so small classes stay readable."""
    idx = list(range(len(CLASS_NAMES)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(idx)
    ax.set_yticks(idx)
    ax.set_xticklabels(CLASS_NAMES, rotation=90)
    ax.set_yticklabels(CLASS_NAMES)
    for i in idx:
        for j in idx:
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} - Confusion matrix (test set, counts; colour = row-normalised)")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    Path(cm_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_png, dpi=150)
    plt.show()

# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_curves(log_csv, title, out_png):
    """
    Accuracy and loss per epoch from the CSVLogger file.

    Reads the log rather than the in-memory History object so the plot always
    matches what is on disk, including a run resumed with append=True.
    """
    log = pd.read_csv(log_csv)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].plot(log["epoch"], log["accuracy"], label="train")
    ax[0].plot(log["epoch"], log["val_accuracy"], label="val")
    ax[0].set_title(f"{title} - Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(log["epoch"], log["loss"], label="train")
    ax[1].plot(log["epoch"], log["val_loss"], label="val")
    ax[1].set_title(f"{title} - Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
