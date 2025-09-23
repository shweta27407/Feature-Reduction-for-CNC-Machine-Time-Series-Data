# Feature Reduction in CNC Machine Time-Series Data

This project explores different methods to **reduce features**, **evaluate predictive power**, and **interpret model outputs** on CNC machine time-series data.  

---

## 🔹 Feature Reduction

### 1. Principal Component Analysis (PCA)
- Applies linear dimensionality reduction.  
- Retains 95% variance (configurable).  
- Produces:
  - `pca_latent.csv` containing reduced features (`PC1..PCk`) + target
  - `pca_scree.png` showing explained variance across components  

---

### 2. Dense Autoencoder (DenseAE)
- Non-linear compression using a fully connected autoencoder.  
- Encoder bottleneck (`latent` layer) provides reduced features.  
- Produces:
  - `dense_latent.csv` containing latent features (`latent_0..latent_n`) + target  
  - Saved models: `dense_autoencoder.h5`, `dense_encoder.h5`  

---

### 3. LSTM-CNN Autoencoder
- Temporal autoencoder combining convolution and recurrent layers.  
- Learns latent representations of sequential windows.  
- Produces:
  - `lstm_cnn_latent.csv` with latent features aligned to the target  

---

## 🔹 Evaluation of Reduced Features

For each reduced representation, an **LSTM Regressor** is trained to predict the target variable.  

### 1. PCA Latent Evaluation
- Input: `pca_latent.csv`  
- Uses context window (`Config.context_dense_pca`) to build sequences  
- Produces:
  - Metrics (R², MAE, RMSE)  
  - `pca_true_vs_pred.png` (True vs Predicted plot)  

---

### 2. DenseAE Latent Evaluation
- Input: `dense_latent.csv`  
- Uses context window (`Config.context_dense_pca`)  
- Produces:
  - Metrics (R², MAE, RMSE)  
  - `denseae_true_vs_pred.png`  

---

### 3. LSTM-CNN Latent Evaluation
- Input: `lstm_cnn_latent.csv`  
- Uses shorter context window (`Config.context_lstm_cnn`)  
- Produces:
  - Metrics (R², MAE, RMSE)  
  - `lstm_cnn_true_vs_pred.png`  

---

## 🔹 Interpretability

### 1. PCA Loadings
- Calculates **loading scores** to quantify how much each original feature contributes to each principal component.  
- Aggregated importance provides a ranking of features.  
- Produces:
  - `pca_top30_loadings.png` — Top 30 most important features  
  - `pca_bottom30_loadings.png` — Bottom 30 least important features  

---

### 2. SHAP for Dense Autoencoder
- Uses **SHAP GradientExplainer** on the encoder’s latent space.  
- Attributes contributions of original features to learned latent dimensions.  
- Produces:
  - `denseae_top30.png` — Top 30 features shaping the latent space  
  - `denseae_bottom30.png` — Bottom 30 least important features  

---

## ✅ Summary of Outputs

- **Latent CSVs**  
  - `pca_latent.csv`, `dense_latent.csv`, `lstm_cnn_latent.csv`

- **Evaluation Plots**  
  - `pca_true_vs_pred.png`, `denseae_true_vs_pred.png`, `lstm_cnn_true_vs_pred.png`

- **Interpretability Plots**  
  - `pca_top30_loadings.png`, `pca_bottom30_loadings.png`  
  - `denseae_top30.png`, `denseae_bottom30.png`  

---

**Author:** Shweta Avinash Bambal  
**Purpose:** Master’s Project — Feature Reduction, Evaluation, and Interpretability for CNC Machine Time-Series