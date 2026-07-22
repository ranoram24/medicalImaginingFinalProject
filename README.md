# Brain Tumor Classification from MRI Scans

Final project for a Medical Image Processing course: CNN-based classification of brain MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

**Authors:** Ran Uram, Shaked Rosenberg

## Dataset

- ~32,000 images aggregated from three open-source Kaggle datasets, deduplicated with perceptual hashing (pHash) to prevent leakage across splits.
- Final dataset: 18,979 unique images (Glioma: 6,329, No Tumor: 5,015, Meningioma: 4,874, Pituitary: 2,761).
- Split 64% / 16% / 20% into train / validation / test.
- A separate undersampled, class-balanced split (11,044 images) was built for Model 8.

## Models

| # | Description |
|---|---|
| 1 | Baseline CNN, lr=0.0001 |
| 2 | Baseline CNN, lower lr (0.00005), stricter early stopping |
| 3 | Baseline CNN, extended class weighting (Meningioma 1.5x, Pituitary 2.0x) |
| 4 | DenseNet-121 (frozen), transfer learning |
| 5 | DenseNet-121, fine-tuned last 50 layers |
| 6 | Custom CNN at 512x512 resolution |
| 7 | EfficientNetB0 (frozen), transfer learning at 512x512 |
| 8 | Baseline CNN on undersampled, class-balanced dataset |

## Results

| Model | Val. Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| Model 1 | 0.9644 | 0.9666 | 0.9964 |
| Model 2 | 0.9193 | 0.9272 | 0.9898 |
| Model 3 | 0.9704 | 0.9734 | 0.9971 |
| Model 4 | 0.9447 | 0.9477 | 0.9953 |
| Model 5 | 0.9697 | 0.9711 | 0.9982 |
| Model 6 | 0.9658 | 0.9668 | 0.9974 |
| **Model 7** | **0.9733** | **0.9743** | **0.9988** |

Model 7 (EfficientNetB0, 512x512) was selected via a weighted multi-criteria ranking and achieved 97% accuracy on the held-out test set. Model 8 achieved 94% accuracy on its own held-out test set, and is evaluated separately since it uses an independently-constructed balanced split rather than the shared main split.

Full methodology and analysis are in the report (`Report.pdf`) and the notebook.

## Repository Structure

```
.
├── ImageProcessingFinalProjectCode_RanUram_ShakedRosenberg.ipynb
├── Brain Tumor Dataset/                    # reference only
├── Brain Tumor MRI Dataset/                # reference only
└── Brain Tumor MRI Multi-Class Dataset/    # reference only
```

The notebook downloads all image data from Google Drive at runtime. The dataset folders are included for reference only, no manual download needed.

## Running the Code

Open the notebook in Google Colab (or any GPU-enabled Jupyter environment) and run all cells. Dependencies and data are fetched automatically.

If you want to train an individual model rather than running the whole notebook, run all the cells up through the Main Data Split first (imports, data download, duplicate removal, and the train/validation/test split). From there, each of the 8 models is fully independent of the others and can be run on its own, in any order.

## Contributions

- **Ran Uram**: deep learning models, transfer-learning architectures, balanced-dataset (undersampling) model, hyperparameter tuning, and the weighted multi-criteria ranking system.
- **Shaked Rosenberg**: dataset aggregation, duplicate detection via perceptual hashing, image normalization and class-weighting strategy, and exploratory data analysis.
