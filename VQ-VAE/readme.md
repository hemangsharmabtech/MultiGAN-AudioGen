
```markdown
# VQ-VAE for Voice Conversion 🎧

This repository contains a Vector Quantized Variational Autoencoder (VQ-VAE) implementation tailored for voice conversion tasks. It leverages Mel spectrograms and MFCC features for learning discrete latent audio representations.

---

## 🧠 Architecture

- **Encoder**: Compresses input spectrograms into a latent space.
- **Codebook**: Vector quantization of latent space to discrete tokens.
- **Decoder**: Reconstructs audio features from quantized representations.
- Trained with **reconstruction loss**, **VQ loss**, and **commitment loss**.

---

## 📁 Directory Structure

```
├── data/
│   ├── mel/                # Mel spectrograms (.npy)
│   └── mfcc/               # MFCC features (.npy)
├── models/
│   ├── vqvae_encoder.pth
│   ├── vqvae_decoder.pth
│   └── vqvae_codebook.pth
├── train_vqvae.py
├── inference_vqvae.py
├── vqvae_model.py
└── README.md
```

---

## 🚀 Training

```bash
python train_vqvae.py
```

This will save the encoder, decoder, and codebook separately under `models/`.

---

## 📦 Loading Model Components

```python
import torch
from vqvae_model import VQVAE

# Define input dimension (e.g., mel + mfcc)
input_dim = 93
model = VQVAE(input_dim=input_dim)

# Load weights
model.encoder.load_state_dict(torch.load("models/vqvae_encoder.pth"))
model.decoder.load_state_dict(torch.load("models/vqvae_decoder.pth"))
model.codebook.load_state_dict(torch.load("models/vqvae_codebook.pth"))

model.eval()
print("VQ-VAE model components loaded successfully ✅")
```

---

## 📊 Dependencies

```bash
pip install torch numpy matplotlib tqdm
```

---

