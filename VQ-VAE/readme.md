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
│   ├── vqvae_codebook.pth     # Saved codebook
│   ├── encoder.joblib         # Saved encoder (joblib)
│   └── decoder.joblib         # Saved decoder (joblib)
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

### Option 1: Load VQVAE Components

```python
import torch
from vqvae_model import VQVAE

# Define the input dimension (e.g., mel + mfcc)
input_dim = 93
model = VQVAE(input_dim=input_dim)

# Load saved weights
model.encoder.load_state_dict(torch.load("models/vqvae_encoder.pth"))
model.decoder.load_state_dict(torch.load("models/vqvae_decoder.pth"))
model.codebook.load_state_dict(torch.load("models/vqvae_codebook.pth"))

model.eval()
print("VQ-VAE model components loaded successfully ✅")
```

### Option 2: Load Encoder and Decoder from Joblib

```python
import torch.nn as nn
import joblib

input_dim = 93

# Load saved encoder
encoder = nn.Sequential(
    nn.Linear(input_dim, 8192),
    nn.ReLU(),
    nn.Linear(8192, 7168),
    nn.ReLU(),
    nn.Linear(7168, 6144),
    nn.ReLU(),
    nn.Linear(6144, 5120),
    nn.ReLU(),
    nn.Linear(5120, 4096),
    nn.ReLU(),
    nn.Linear(4096, 3072),
    nn.ReLU(),
    nn.Linear(3072, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU()
)

encoder.load_state_dict(joblib.load(r"C:\\Users\\cl502_11\\MG\\Models\\VQ-VAE\\Autoencoder\\encoder.joblib"))
encoder.eval()

# Load saved decoder
decoder = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 3072),
    nn.ReLU(),
    nn.Linear(3072, 4096),
    nn.ReLU(),
    nn.Linear(4096, 5120),
    nn.ReLU(),
    nn.Linear(5120, 6144),
    nn.ReLU(),
    nn.Linear(6144, 7168),
    nn.ReLU(),
    nn.Linear(7168, 8192),
    nn.ReLU(),
    nn.Linear(8192, input_dim)
)

decoder.load_state_dict(joblib.load(r"C:\\Users\\cl502_11\\MG\\Models\\VQ-VAE\\Autoencoder\\decoder.joblib"))
decoder.eval()

print("Encoder and Decoder loaded from joblib successfully ✅")
```

---


