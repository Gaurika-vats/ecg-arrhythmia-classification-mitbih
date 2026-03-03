# ECG Arrhythmia Classification using MIT-BIH Dataset

## Project Overview
This project implements an end-to-end deep learning pipeline for automated heartbeat classification using the MIT-BIH Arrhythmia Dataset. The system performs ECG signal preprocessing, heartbeat segmentation around R-peaks, RR interval feature extraction, stratified patient-level data splitting to prevent leakage, class imbalance handling using capped weighted sampling, and training of a hybrid CNN model that fuses morphological and rhythm features for multi-class arrhythmia detection.

## Dataset
The dataset used is the MIT-BIH Arrhythmia Dataset from PhysioNet.
- **Source:** PhysioNet
- **Records:** 48 half-hour two-channel ambulatory ECG recordings
- **Sampling Frequency:** 360 Hz
- **Annotations:** Beat-level annotations provided by cardiologists

The model is trained on standard heartbeat categories following the AAMI EC57 standard grouping:
- **N:** Normal beat (includes Left/Right bundle branch block beats)
- **S:** Supraventricular ectopic beat
- **V:** Ventricular ectopic beat
- **F:** Fusion beat
- **Q:** Unknown beat (removed during preprocessing)

Although the dataset provides two ECG channels per record, this implementation uses only the MLII lead for heartbeat classification.

The dataset is not included in this repository. Users must download it directly from PhysioNet.

## Data Preprocessing

### Signal Segmentation
- ECG signal loading using **wfdb**
- R-peak extraction using annotation files
- Fixed-length window of 0.6 seconds (216 samples at 360 Hz) centered around each R-peak
- Removal of beats at signal boundaries and Q-class beats

### Signal Normalization
- Per-beat z-score normalization applied to each 216-sample window independently to remove amplitude variability across patients and leads

### RR Interval Feature Extraction
For each beat, 6 rhythm context features are extracted alongside the morphological window:

| Feature | Description |
|---|---|
| pre_RR | Samples between previous R-peak and current |
| post_RR | Samples between current R-peak and next |
| ratio | pre_RR / post_RR — captures early/late arrival |
| local_avg_RR | Mean RR over the preceding 5 beats |
| norm_pre_RR | pre_RR normalized by local average |
| norm_post_RR | post_RR normalized by local average |

These features provide rhythm context that morphology alone cannot capture — particularly important for Supraventricular beats (S-class), which arrive early relative to the local rhythm (short pre_RR) and are followed by a near-compensatory pause (longer post_RR).

All RR features are z-score normalized across the full dataset before training.

### Patient-Level Stratified Split
Standard random patient splits risk concentrating S-class beats (which are heavily patient-specific in MIT-BIH) in a single partition. To address this, S-heavy patients (≥50 S-beats) are identified and distributed in a round-robin fashion across splits before randomly assigning remaining patients:

| Split | Patients | Beats |
|---|---|---|
| Train | 30 | 61,954 |
| Validation | 8 | 16,278 |
| Test | 10 | 24,155 |

No patient appears in more than one split. Zero overlap is verified programmatically via assertion checks.

### Class Imbalance Handling
A `WeightedRandomSampler` is used during training with inverse-frequency class weights. Weights are capped at 10× the N-class weight to prevent over-firing on the smallest classes (uncapped, F-class would receive ~77× oversampling).

## Model Architecture

The model is a hybrid 1D CNN with late fusion of RR interval features, designed for single-beat ECG classification.

Each input consists of a morphological segment (216 samples) and a 6-dimensional RR feature vector. The CNN processes the morphological signal, and the learned embedding is concatenated with the RR features before the final classification layer.

```
Morphological branch:
  Input: 1 × 216

  Conv1D(32, kernel=7, padding=3) → BatchNorm → ReLU → MaxPool1D(2)
  Conv1D(64, kernel=5, padding=2) → BatchNorm → ReLU → MaxPool1D(2)
  Conv1D(128, kernel=3, padding=1) → BatchNorm → ReLU

  Global Average Pooling → (batch, 128)

RR branch:
  Input: (batch, 6) — concatenated directly after GAP

Fusion:
  Concat → (batch, 134)
  Dropout(0.4)
  Fully Connected (134 → 4)
```

The decreasing kernel sizes (7 → 5 → 3) enable hierarchical feature extraction: the first layer captures broad waveform morphology (QRS complex shape), subsequent layers refine finer temporal patterns. Global Average Pooling replaces flattening to reduce overfitting and parameter count.

## Training Setup

| Hyperparameter | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| LR Scheduler | CosineAnnealingLR (T_max=60, eta_min=1e-6) |
| Batch Size | 64 (train), 128 (val/test) |
| Max Epochs | 60 |
| Early Stopping | Patience=7 on validation Macro F1 |
| GPU | NVIDIA T4 (Google Colab) |

Early stopping monitors validation **Macro F1**.

Random seeds and deterministic settings are fixed for reproducibility.

## Results

Evaluated on a held-out test set with strict patient-level separation.

### Overall Metrics

| Metric | Value |
|---|---|
| Overall Accuracy | 75.79% |
| Macro F1 Score | 0.4998 |

### Class-wise Performance

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| N (Normal) | 0.9661 | 0.7523 | 0.8459 |
| S (Supraventricular ectopic) | 0.1622 | 0.2422 | 0.1943 |
| V (Ventricular ectopic) | 0.5508 | 0.9384 | 0.6942 |
| F (Fusion) | 0.1581 | 0.8150 | 0.2649 |

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

## Key Insights

- **Normal beats (N)** achieve strong precision (0.966) reflecting the model's reliable identification of the dominant class.
- **Ventricular beats (V)** achieve excellent recall (0.938), which is clinically significant as missed V-beats carry higher risk.
- **Supraventricular beats (S)** remain the hardest class. Single-beat morphology alone is insufficient to distinguish S from N due to their visual similarity. Adding RR interval features improved S-class F1, the rhythm context (shorter pre-RR, near-compensatory post-RR) provides discriminating signal that morphology cannot.
- **Macro F1 (0.499) is the primary reported metric** rather than overall accuracy (75.79%), as accuracy is misleading under class imbalance.
- **Patient-wise stratified splitting** was critical: naive random patient splits concentrated S-heavy patients in a single partition. Explicit distribution of S-heavy patients (≥50 S-beats) across all splits produced more representative and reliable evaluation.
- The performance ceiling for this architecture on this dataset is largely a **data constraint**: MIT-BIH contains only 48 patients, S-beats are concentrated in 2–3 patients, and patient-wise evaluation inherently limits generalization measurement.

## How to Run

**Run on Google Colab (Recommended)**
1. Open the notebook in Google Colab
2. Download the MIT-BIH dataset from PhysioNet
3. Upload the dataset to your Google Drive
4. Update `DATA_DIR` in Cell 4 to your Drive path
5. Run all cells sequentially

A pretrained model checkpoint is provided at `models/best_model.pth`. Load it directly for evaluation without retraining:

```python
model = ECGModel(num_classes=4, rr_features=6).to(device)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()
```

**Note:** The project was developed and tested using Python 3.12.12.
