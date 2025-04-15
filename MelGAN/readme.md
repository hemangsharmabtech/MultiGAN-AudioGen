# 🎵 MelGAN Voice Conversion

This project implements a simple GAN-based architecture for **voice conversion**, specifically using **Mel spectrogram latent vectors** to generate audio-like outputs. Inspired by MelGAN-style training, this model is designed for rapid prototyping and educational use.

## 📁 Project Structure

```
MelGAN-Voice-Convert/
│
├── models/
│   ├── generator_epoch_50.pt
│   └── discriminator_epoch_50.pt
│
├── src/
│   ├── train.py               # Training script for Generator and Discriminator
│   └── load_model.py          # Load trained models and generate audio
│
├── data/
│   ├── MW_latent.npy          # Input latent vectors for generator
│   └── Real/                  # Real audio vectors for discriminator
│
├── README.md
└── requirements.txt
```

---

## 🧠 GAN Architecture

- **Generator**: Converts 128-dimensional Mel-spectrogram latent vector to 1024-dimensional audio-like feature.
- **Discriminator**: Distinguishes between generated and real audio features.

Both models are trained using Binary Cross-Entropy Loss and the standard GAN training loop.

---

## 🚀 Load Trained Models

To use the trained Generator and Discriminator:

### 📦 Dependencies

```bash
pip install torch numpy
```

### 📜 Load Model Script

```python
from src.load_model import load_trained_models

GEN_PATH = "models/generator_epoch_50.pt"
DISC_PATH = "models/discriminator_epoch_50.pt"
LATENT_DIM = 128      # Replace with your actual latent dimension
AUDIO_LEN = 1024      # Replace with your actual output audio length

# Load models
generator, discriminator = load_trained_models(
    GEN_PATH, DISC_PATH, LATENT_DIM, AUDIO_LEN
)

# Generate a sample output
import torch
latent = torch.randn(1, LATENT_DIM)  # Random input
generated_audio = generator(latent)
print("Generated output shape:", generated_audio.shape)
```

---

## 📈 Training

To retrain the models or continue training, run:

```bash
python src/train.py
```

Make sure your data is structured and paths are set correctly in the script.

---

## 🧊 License

This project is open-source under the MIT License.

---

## 💬 Acknowledgements

- PyTorch Team  
- MelGAN (original paper)  
- Audio research community

---

🎧 Built with love for audio experimentation and deep learning.
