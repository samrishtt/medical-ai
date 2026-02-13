
# Milk10k Training Instructions

Since you don't have a local GPU, follow these steps to train the model using **Kaggle** (Free GPU).

## 1. Prepare Data
1.  Go to [ISIC Challenge - MILK10k](https://challenge.isic-archive.com/landing/milk10k/).
2.  Log in and download the data.
3.  Organize the data on your computer like this (or just keep the zip):
    ```
    milk10k/
    ├── train/           # Contains all images (clinical + dermoscopic)
    ├── train.csv        # Metadata with columns: image_name, diagnosis, etc.
    └── val.csv          # (Optional) Validation split
    ```

## 2. Upload to Kaggle
1.  Go to [Kaggle Kernels](https://www.kaggle.com/code).
2.  Click **New Notebook**.
3.  **File -> Import Notebook**: Upload `notebooks/milk10k_train.ipynb`.
4.  **Add Data**:
    - Click **Add Data** (right sidebar).
    - Upload your `milk10k` dataset.
5.  **Enable GPU**:
    - Settings -> Accelerator -> **GPU T4 x2** (or P100).
6.  **Run**:
    - Execute the cells.
    - The notebook will install dependencies and start training.

## 3. Monitor & Submission
- The notebook saves checkpoints to `outputs/checkpoints`.
- Download the best checkpoint.
- Use the inference script (to be created if needed) to generate `submission.csv`.

## Local Code Changes
I have updated your local codebase to support this:
- **`src/data/milk10k_dataset.py`**: Handles dual-image loading (Clinical + Dermoscopic).
- **`src/models/tam_vit.py`**: Updated `TAMViT` to accept 6-channel input (stacked images).
- **`configs/milk10k.yaml`**: Configuration for the experiment.
- **`notebooks/milk10k_train.ipynb`**: Ready-to-go notebook for Kaggle.
