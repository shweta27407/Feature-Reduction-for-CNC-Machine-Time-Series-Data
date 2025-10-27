# Feature Reduction in CNC Machine Time-Series Data

This project explores different methods to **reduce features**, **evaluate predictive power**, and **interpret model outputs** on CNC machine time-series data.  

---

## ⚙️ Configuration

The file **`config.py`** controls key experiment settings:

- **`data_path`** → Path to the dataset file (you can switch between datasets by editing this).  
- **`split`** → Train/validation/test split ratio (default: `0.6, 0.2, 0.2`).  
- **`window_size`** → Sequence length for time-series models.  

Update these values in `config.py` to adapt the experiments to different datasets or configurations.

---

## ⚙️ Installation

Before running the project, make sure to install all dependencies.  
You can install them directly from the `requirements.txt` file:

```
pip install -r requirements.txt
```

---

## 🔹 Feature Reduction

### 1. Principal Component Analysis (PCA)
- Applies linear dimensionality reduction.  
- Retains 98% variance (configurable).  
- Produces:
  - `pca_latent.csv` containing reduced features (`PC1..PCk`) + target
  - `pca_scree.png` showing explained variance across components 

#### Run PCA Feature Reduction

**Default (uses paths & dataset split ratio from `config.py`)**

```
python main.py --model pca_model
``` 

---

### 2. Dense Autoencoder
- Non-linear compression using a fully connected autoencoder.  
- Encoder bottleneck (`latent` layer) provides reduced features.  
- Produces:
  - `dense_latent.csv` containing latent features (`latent_0..latent_n`) + target  
  - Saved models: `dense_autoencoder.h5`, `dense_encoder.h5` 

#### Run Autoencoder Feature Reduction

```
python main.py --model dense_ae
```  

---

### 3. LSTM-CNN Autoencoder
- Temporal autoencoder combining convolution and recurrent layers.  
- Learns latent representations of sequential windows.  
- Produces:
  - `lstm_cnn_latent.csv` with latent features aligned to the target 

#### Run LSTM+CNN Autoencoder Feature Reduction


```
python main.py --model lstm_cnn
```   

---

## 🔹 Evaluation of Reduced Features

For each reduced representation, an **LSTM Regressor** is trained to predict the target variable.  

### 1. PCA Latent Evaluation
- Input: `pca_latent.csv`  
- Uses context window (Window_size = 60, as it produced the best accuracy score in our experimentation) to build sequences  
- Produces:
  - Metrics (R², MAE, RMSE)  
  - `pca_true_vs_pred.png` (True vs Predicted plot)  

#### Command for Evaluation

**Default (uses paths & dataset split ratio from `config.py`)**

```
python -m evaluation.lstm_regressor --latent_csv output/pca_latent.csv --model_type pca
```   
---

### 2. DenseAE Latent Evaluation
- Input: `dense_latent.csv`  
- Uses context window (Window_size = 60)
- Produces:
  - Metrics (R², MAE, RMSE)  
  - `denseae_true_vs_pred.png`  

#### Command for Evaluation

```
python -m evaluation.lstm_regressor --latent_csv output/dense_latent.csv --model_type dense_ae
```  

---

### 3. LSTM-CNN Latent Evaluation
- Input: `lstm_cnn_latent.csv`  
- Uses shorter context window (Window_size = 10)  as it produced best accuracy score.  
- Produces:
  - Metrics (R², MAE, RMSE)  
  - `lstm_cnn_true_vs_pred.png`  

#### Command for Evaluation

```
python -m evaluation.lstm_regressor --latent_csv output/lstm_cnn_latent.csv --model_type lstm_cnn
```  


---

## 🔹 Interpretability

### 1. PCA Loadings
- Calculates **loading scores** to quantify how much each original feature contributes to each principal component.  
- Aggregated importance provides a ranking of features.  
- Produces:
  - `pca_top30_loadings.png` — Top 30 most important features  
  - `pca_bottom30_loadings.png` — Bottom 30 - least important features  

#### Command for PCA Loadings Plot

```
python -m interpretability.loadings --data_path datasets/DMC2_S_CP2.csv
``` 

---

### 2. SHAP for Dense Autoencoder
- Uses **SHAP GradientExplainer** on the encoder’s latent space.  
- Attributes contributions of original features to learned latent dimensions.  
- Produces:
  - `denseae_top30.png` — Top 30 features shaping the latent space  
  - `denseae_bottom30.png` — Bottom 30 - least important features  

#### Command for SHAP Plot

```
python -m interpretability.shap_ae --model_path output/dense_autoencoder.h5
``` 

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

## 📊 Results
- Reduced original feature set from **52 → ~25** latent features using PCA and Autoencoder Architectures.  
- Using the reduced features, Supervised LSTM-CNN achieved **R² ≈ 0.8**, exceeding the original full-feature performance (≈ 0.7).  

## ✨ Significance
- Identified the **most important sensor features**, enabling prioritized monitoring and maintenance in CNC manufacturing.  
- Highlighted redundant and less useful sensors, allowing data collection and storage costs to be reduced.  
- Improved model efficiency: less input data → faster training/inference, lower memory and computational demands.  
- Enhanced interpretability and trust in model decisions, which is critical for industrial applications and deployment in sensor systems.  


