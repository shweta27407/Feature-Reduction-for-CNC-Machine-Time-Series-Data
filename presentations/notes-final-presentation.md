Vanilla AE :

🔹 What is a vanilla autoencoder?
	•	Vanilla AE = the simplest form of an autoencoder.
	•	Structure:
	•	Input layer → Encoder (Dense layers) → Latent bottleneck → Decoder (Dense layers) → Output layer
	•	It is fully connected (Dense), no CNNs, no LSTMs, no fancy tricks (like variational loss, attention, adversarial objectives).
	•	Activation functions are usually ReLU (encoder/decoder hidden layers) and Linear in the output (to reconstruct continuous data).
	•	Trained with MSE loss between input and reconstruction.

This is the “basic” AE introduced in early research papers — that’s why it’s called vanilla (plain/basic).

🔹 Why it’s vanilla:
	•	Fully-connected Dense encoder–decoder ✔️
	•	Symmetric structure ✔️
	•	Latent bottleneck ✔️
	•	Reconstruction loss (MSE) ✔️
	•	No advanced components like attention, CNN, LSTM, adversarial loss ❌

So your code = vanilla autoencoder with dropout regularization. ✅

⸻
