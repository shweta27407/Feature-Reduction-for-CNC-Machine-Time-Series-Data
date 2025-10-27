# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # -------------------
    # Dataset parameters
    # -------------------
    data_path: str = ("/Users/shwetabambal/Desktop/feature reduction project/p3-feature-reduction-in-time-series/datasets/DMC2_S_CP2.csv")
    window_size: int = 60
    split: tuple = (0.6, 0.2, 0.2)  # train, val, test (must sum to 1.0)

    # -------------------
    # Evaluation parameters
    # -------------------
    context_lstm_cnn: int = 10  # context window for latent sequences from LSTM-CNN
    context_dense_pca: int = 60  # context window for latent sequences from DenseAE and PCA