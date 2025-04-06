# VQ-VAE for Voice Conversion 🎧

This repository contains a **Vector Quantized Variational Autoencoder (VQ-VAE)** implementation tailored for voice conversion tasks. It leverages **Mel spectrograms** and **MFCC features** to learn discrete latent representations of audio data.

---

## 🧠 Architecture

- **Encoder**: Compresses input spectrograms into a latent space.
- **Codebook**: Performs vector quantization to convert continuous latent space into discrete tokens.
- **Decoder**: Reconstructs audio features from quantized representations.
- The model is trained using a combination of:
  - **Reconstruction Loss**
  - **VQ Loss**
  - **Commitment Loss**

---

## 📁 Directory Structure

```
project-root/
├── data/
│   ├── mel/                   # Mel spectrograms (.npy)
│   └── mfcc/                  # MFCC features (.npy)
├── models/
│   ├── vqvae_encoder.pth      # Saved encoder
│   ├── vqvae_decoder.pth      # Saved decoder
│   └── vqvae_codebook.pth     # Saved codebook
├── train_vqvae.py             # Training script
├── inference_vqvae.py         # Inference/demo script
├── vqvae_model.py             # VQ-VAE model definition
└── README.md
```

---

## 🚀 Training

Run the following command to train the model:

```bash
python train_vqvae.py
```

After training, the encoder, decoder, and codebook will be saved separately inside the `models/` folder.

---

## 📦 Loading Model Components

```python
import torch
import torch.nn as nn
from vqvae_model import VQVAE

# Define the input dimension (e.g., mel + mfcc)
input_dim = 93

# Instantiate model
model = VQVAE(input_dim=input_dim)

# Load model weights
model.encoder.load_state_dict(torch.load("models/vqvae_encoder.pth"))
model.decoder.load_state_dict(torch.load("models/vqvae_decoder.pth"))
model.codebook.load_state_dict(torch.load("models/vqvae_codebook.pth"))

model.eval()
print("VQ-VAE model components loaded successfully ✅")
```

---

## 📊 Dependencies

Install the required dependencies using:

```bash
pip install torch numpy matplotlib tqdm joblib
```

---
