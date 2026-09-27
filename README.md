# Noninvasive Glucose Prediction from Fitbit Data

Predicting blood glucose from wearable signals, with no finger pricks or skin sensors at prediction time.

This project uses heart rate variability (HRV) and blood-oxygen signals recorded by a **Fitbit** to predict glucose levels measured by a **continuous glucose monitor (CGM)**. The CGM serves as the ground truth during training; the goal is a model that estimates glucose from the Fitbit's signals alone.

Five modeling approaches are compared, from a linear baseline to a stacked ensemble of a GRU neural network and a random forest.

## Highlights

- **Best model:** a stacked ensemble (GRU + Random Forest, combined with Ridge regression) reaching **R² = 0.77** and **RMSE ≈ 6.1 mg/dL** on held-out data.
- **Clinical accuracy:** **98.9%** of ensemble predictions fall in Zone A of the Clarke Error Grid, with **no predictions in the dangerous Zones C or E**.
- **Sequence models matter:** recurrent networks that see the preceding 90 minutes of HRV data far outperform models that treat each reading independently (R² 0.64 to 0.77 vs. 0.13 to 0.22).

## Data

| | |
|---|---|
| Glucose source (target) | Continuous glucose monitor, in mmol/L (converted to mg/dL for the Clarke Error Grid) |
| Wearable source (inputs) | Fitbit |
| Participants | 4 |
| Recording sessions | 45 (mostly overnight, plus a few morning and afternoon sessions) |
| Sampling interval | 5 minutes |
| Total samples | ~3,900 time points |

**Input features**

| Feature | Description |
|---|---|
| `rmssd` | Root mean square of successive differences between heartbeats, a measure of short-term HRV |
| `lf` | Low-frequency power of HRV (sympathetic and parasympathetic activity) |
| `hf` | High-frequency power of HRV (parasympathetic activity) |
| `Infrared to Red Signal Ratio` | Fitbit's blood-oxygen (SpO₂) variation signal |
| `Height`, `Weight`, `Age`, `Gender` | Participant demographics (used by the linear, random forest and ensemble models) |

Each session is stored as a CSV with the columns `timestamp, rmssd, lf, hf, Infrared to Red Signal Ratio, glucose`, where Fitbit readings have been aligned to CGM readings on the same 5-minute timestamps.

> **Note:** The dataset is not included in this repository and is not publicly available, to protect participant confidentiality. As a result, the notebooks cannot be run on your own computer (see [Running the Code](#running-the-code)).

## Method

### Preprocessing

1. **Alignment:** Fitbit HRV and SpO₂ readings are matched to CGM glucose readings at 5-minute intervals.
2. **Outlier handling:** values with a Z-score beyond a threshold are replaced with the feature mean.
3. **Standardization:** features and glucose are scaled with `StandardScaler`; predictions are inverse-transformed back to mmol/L for evaluation.
4. **Windowing:** for the sequence models, each prediction uses a sliding window of the previous **18 time steps (90 minutes)** of Fitbit data to predict the glucose value at the next step. Windows never cross session boundaries.

### Models

| Notebook | Model | Input |
|---|---|---|
| `LinearRegression.ipynb` | Linear regression (baseline) | Single time point: HRV + SpO₂ + demographics |
| `RandomForest.ipynb` | Random forest (200 trees) | Current values plus 17 lagged values of each HRV/SpO₂ feature, plus demographics |
| `FinalLSTM.ipynb` | LSTM (64 units) → 3 dense layers (64) with dropout | 90-minute window of HRV + SpO₂ |
| `GRUModel.ipynb` | GRU (64 units) → 3 dense layers (64) with dropout | 90-minute window of HRV + SpO₂ |
| `EnsembleModelFinal.ipynb` | Stacked ensemble: GRU + Random Forest, combined by Ridge regression | GRU gets the HRV window; RF gets the flattened window + demographics |

Neural networks are trained with the Adam optimizer and MSE loss for 100 epochs (batch size 8) using TensorFlow/Keras. The ensemble notebook also runs 5-fold time-series cross-validation.

### Evaluation

- **Regression metrics:** RMSE, MAE, and R².
- **Clarke Error Grid Analysis:** the standard clinical tool for judging glucose-estimate accuracy. Zone A is clinically accurate, Zone B is benign, and Zones C–E could lead to incorrect treatment decisions.
- **Deployment metrics:** inference time per sample and model size, to gauge feasibility on a wearable or phone.

## Results

| Model | RMSE (mmol/L) | RMSE (mg/dL) | R² | Clarke Zone A | Zone B | Zone D | Zones C + E |
|---|---|---|---|---|---|---|---|
| Linear Regression | 0.73 | 13.2 | 0.13 | 86.6% | 9.3% | 4.1% | 0% |
| Random Forest | 0.62 | 11.2 | 0.22 | 89.7% | 6.5% | 3.8% | 0% |
| GRU | 0.39 | 7.1 | 0.64 | 96.3% | 2.4% | 1.3% | 0% |
| LSTM | 0.38 | 6.9 | 0.72 | 97.9% | 1.0% | 1.1% | 0% |
| **Ensemble (GRU + RF → Ridge)** | **0.34** | **6.1** | **0.77** | **98.9%** | 0.6% | 0.6% | 0% |

**Feature importance.** In the random forest, HRV features carried most of the predictive signal: RMSSD (28%), LF (23%), SpO₂ lags (19%), and HF (17%), with demographics contributing less.

**Speed and size.** Linear regression runs in about 0.7 ms per sample (34 bytes of parameters). The random forest takes about 42 ms per sample and is about 32 MB on disk. The LSTM and GRU run in roughly 50–70 ms per sample on a laptop CPU.

## Repository Structure

```
NoninvasiveGlucosePrediction/
├── LinearRegression.ipynb     # Baseline linear model
├── RandomForest.ipynb         # Random forest with lagged features + feature importance
├── FinalLSTM.ipynb            # LSTM sequence model
├── GRUModel.ipynb             # GRU sequence model + correlation heatmap
├── EnsembleModelFinal.ipynb   # GRU + RF stacked with Ridge regression, time-series CV
└── README.md
```

## Running the Code

**These notebooks cannot be run locally.** The participant data they depend on is confidential and is not shared, so the data-loading steps will fail outside the original research environment.

The notebooks are provided so you can review the full methodology: preprocessing, model architectures, training, and evaluation. Each notebook is saved with its outputs, so you can see the printed metrics, training logs, and plots directly on GitHub without running anything.

The code was developed with Python 3.11 using `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, and `seaborn`.

## Acknowledgments

Thanks to the participants who wore both a Fitbit and a CGM and shared their data for this study.
