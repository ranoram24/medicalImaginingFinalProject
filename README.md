# Brain Tumor Classification from MRI Scans

Final project for a Medical Image Processing course: CNN-based classification of brain MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

**Authors:** Ran Uram (206661886), Shaked Rosenberg (209547033)

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
| Model 1 | 0.9513 | 0.9559 | 0.9946 |
| Model 2 | 0.9621 | 0.9660 | 0.9958 |
| Model 3 | 0.9588 | 0.9634 | 0.9963 |
| Model 4 | 0.9434 | 0.9451 | 0.9943 |
| Model 5 | 0.9621 | 0.9630 | 0.9970 |
| Model 6 | 0.9717 | 0.9731 | 0.9977 |
| **Model 7** | **0.9740** | **0.9755** | **0.9986** |

Model 7 (EfficientNetB0, 512x512) was selected via a weighted multi-criteria ranking and achieved 98% accuracy on the held-out test set. Model 8 achieved 95% accuracy on its own held-out test set.

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

## Contributions

- **Ran Uram**: deep learning models, transfer-learning architectures, balanced-dataset (undersampling) model.
- **Shaked Rosenberg**: dataset aggregation, duplicate detection via perceptual hashing, image normalization and class-weighting strategy.
