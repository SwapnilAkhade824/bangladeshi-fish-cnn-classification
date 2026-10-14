# Bangladeshi Freshwater Fish Image Classification Using CNN and Transfer Learning

## Group Members
1. Swapnil Santosh Akhade
2. Mansi Somani
3. Vivek Singh

## Dataset
Bangladeshi Freshwater Fish Image Dataset (BFF-15)

- Source: https://www.kaggle.com/datasets/theshahidul/bangladeshi-freshwater-fish-image-dataset-bff-15
- License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- Uploaded by: Md. Shahidul Islam Prodhan
- Committed under `data/raw/` in this repository with attribution.

See `docs/dataset_source.md` for full details.

## Notebooks
| Notebook | Purpose | Owner |
|---|---|---|
| `notebooks/01_custom_cnn.ipynb` | Load BFF-15, preprocess, augment, train CNN from scratch, save checkpoints + logs + metrics into `custom_cnn/` | Member 2 |
| `notebooks/02_transfer_learning.ipynb` | Load BFF-15 with same split and class mapping, fine-tune pretrained model, save into `transfer_learning/` | Member 3 |
| `notebooks/03_comparison.ipynb` | Read both folders, compare models via accuracy, precision, recall, F1, confusion matrices | Member 1 |

## Folder Layout
- `custom_cnn/` — all artifacts for the from-scratch CNN
  - `logs/` — per-epoch training history CSV (resume + comparison)
  - `metrics/` — final test metrics (accuracy, precision, recall, F1)
  - `plots/` — training curves and confusion matrices
  - `best_model/` — best saved checkpoint
- `transfer_learning/` — same structure for the pretrained model
- `data/raw/` — local raw BFF-15 dataset (not committed)

## Workflow
1. Place BFF-15 into `data/raw/`.
2. Member 2 runs `01_custom_cnn.ipynb`.
3. Member 3 runs `02_transfer_learning.ipynb`.
4. Member 1 runs `03_comparison.ipynb`.

Both model notebooks must use the same class mapping and the same train/validation/test split.

## Environment
```bash
pip install -r requirements.txt
