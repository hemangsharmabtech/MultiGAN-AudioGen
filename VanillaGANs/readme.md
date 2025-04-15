# GANs for Voice Feature Generation 🎤

This repository contains a **Generative Adversarial Network (GAN)** model designed for generating realistic audio features such as Mel spectrograms and MFCCs for voice conversion or synthesis tasks.

---

## 🧠 Architecture

- **Generator**: Takes in a random latent vector (noise) and outputs audio feature representations.
- **Discriminator**: Attempts to distinguish between real and generated audio features.
- The model is trained adversarially using:
  - **Generator Loss** (tries to fool the discriminator)
  - **Discriminator Loss** (tries to correctly classify real and fake inputs)

---

## 📁 Directory Structure

```
project-root/
├── data/
│   ├── mel/                    # Mel spectrograms (.npy)
│   └── mfcc/                   # MFCC features (.npy)
├── models/
│   ├── generator.pth           # Trained generator model
│   └── discriminator.pth       # Trained discriminator model
├── train_gan.py                # Training script
├── inference_gan.py            # Inference/demo script
├── gan_model.py                # GAN model definition
└── README.md
```

---

## 🚀 Training

Run the following to train the GAN model:

```bash
python train_gan.py
```

After training, the generator and discriminator weights will be saved under the `models/` directory.

---

## 📦 Loading the Generator

```python
import torch
import torch.nn as nn
from gan_model import Generator

# Define dimensions
latent_dim = 100
output_dim = 93  # e.g., mel + mfcc combined

# Initialize generator architecture
generator = Generator(latent_dim=latent_dim, output_dim=output_dim)

# Load weights
generator.load_state_dict(torch.load("models/generator.pth"))
generator.eval()

print("Generator loaded successfully ✅")
```

---


## 🧪 Inference Example

```python
import torch

# Generate random noise
z = torch.randn(1, latent_dim)

# Generate fake audio features
fake_features = generator(z)
print(fake_features.shape)
```
