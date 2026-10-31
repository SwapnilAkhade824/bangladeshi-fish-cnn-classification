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

## Results Website (GitHub Pages)
A static results/report page lives in `docs/` and renders from `docs/data/*.json`
(exported by `notebooks/03_comparison.ipynb`).

- Enable it once: repo **Settings → Pages → Build and deployment → Source: Deploy from a branch → `main` → `/docs` → Save**.
- Live URL: https://swapnilakhade824.github.io/bangladeshi-fish-cnn-classification/
- To refresh after re-training: re-run notebook 03, copy `outputs/website/results.json`,
  `outputs/comparison/{final_summary.json,mcnemar.json,comparison_table.csv}` into `docs/data/`, then commit.

## Notebooks
| Notebook | Purpose | Owner |
|---|---|---|
| `notebooks/00_dataset_setup.ipynb` | Scan BFF-15, remove byte-identical duplicates, build the frozen stratified 70/15/15 split, define the shared `tf.data` pipeline | Member 1 (Swapnil) |
| `notebooks/01_custom_cnn.ipynb` | Load BFF-15, preprocess, augment, train CNN from scratch, save checkpoints + logs + metrics into `custom_cnn/` | Member 3 (Vivek) |
| `notebooks/02_transfer_learning.ipynb` | Load BFF-15 with same split and class mapping, fine-tune pretrained model, save into `transfer_learning/` | Member 2 (Mansi) |
| `notebooks/03_comparison.ipynb` | Read both folders, compare models via accuracy, precision, recall, F1, confusion matrices | Member 1 (Swapnil) |

## Folder Layout
- `custom_cnn/` — all artifacts for the from-scratch CNN
  - `logs/` — per-epoch training history CSV (resume + comparison)
  - `metrics/` — final test metrics (accuracy, precision, recall, F1)
  - `plots/` — training curves and confusion matrices
  - `best_model/` — best saved checkpoint
- `transfer_learning/` — same structure for the pretrained model
- `data/raw/` — the BFF-15 dataset (~40 MB), committed here with CC BY 4.0 attribution
- `data/splits/` — the frozen `train.csv` / `val.csv` / `test.csv`, paths stored relative to the repo root
- `outputs/` — comparison table, side-by-side plots and `website/results.json` from notebook 03
- `data_utils.py` — shared scanning, de-duplication, splitting, augmentation and `tf.data` pipeline
- `eval_utils.py` — shared test evaluation, metric export, training-curve plots and the resumable-training callback

## Workflow
1. Place BFF-15 into `data/raw/` (skip if already committed).
2. Run `00_dataset_setup.ipynb` once to (re)generate `data/splits/*.csv`.
3. Member 3 runs `01_custom_cnn.ipynb`.
4. Member 2 runs `02_transfer_learning.ipynb`.
5. Member 1 runs `03_comparison.ipynb`.

Both model notebooks must use the same class mapping and the same train/validation/test split.

## Running on Google Colab
- Upload the repo (or connect GitHub) so `data_utils.py` and `eval_utils.py` sit next to `notebooks/`.
- Runtime → Change runtime type → **T4 GPU**, before running any cell.
- Notebooks 01 and 02 save `best_model/last_state.kerasresumable` after every epoch, so a
  dropped session can be resumed by re-running the notebook; delete that file to force a fresh run.
- Notebook 00 must be run first if the committed split CSVs do not match the uploaded dataset.

## Environment
```bash
pip install -r requirements.txt
