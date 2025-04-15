# 🎵 CycleGAN for Voice Latent Space Translation

This project implements a **CycleGAN** for translating **latent features** (like Mel Spectrograms or MFCCs) between two domains — for example, from your voice to a singer's voice (like KK) and vice versa.

Built with **PyTorch**, it includes:
- Generator and Discriminator architecture
- Adversarial loss training
- Model checkpoint saving
- Loss visualization
- Easy model reloading

---

## 🚀 Project Structure

- `Generator`: Translates latent vector A → B or B → A
- `Discriminator`: Classifies real vs fake in each domain
- Trains two generators (`G_A2B`, `G_B2A`) and two discriminators (`D_A`, `D_B`)
- Uses `.npy` feature files for both input and real samples

---

## 🔧 Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/VoiceLatent-CycleGAN.git
cd VoiceLatent-CycleGAN
```

2. **Install dependencies:**

```bash
pip install torch numpy matplotlib tqdm
```

3. **Prepare your `.npy` feature files** in appropriate folders and update paths accordingly.

---

## 🧠 Load Trained Models & Inference Code

```python
import os
import torch
import torch.nn as nn
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best params used in training (make sure these match!)
latent_dim = 1024
hidden_dim = 1476
dropout = 0.49

# Define Generator (same architecture used during training)
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# Paths (update these as per your directory structure)
MODEL_PATH = r"path/to/save/CycleGANs/G_A2B.pth"
INPUT_FEATURE_PATH = r"path/to/test_input.npy"

# Load model
G_A2B_loaded = Generator(latent_dim, hidden_dim, dropout).to(device)
G_A2B_loaded.load_state_dict(torch.load(MODEL_PATH, map_location=device))
G_A2B_loaded.eval()

# Load test feature
test_input = torch.tensor(np.load(INPUT_FEATURE_PATH), dtype=torch.float32).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    converted_output = G_A2B_loaded(test_input)
    print("Converted output shape:", converted_output.shape)
```

---

## 📌 Notes

- You can change the architecture to make it deeper/more complex.
- Ensure `latent_dim`, `hidden_dim`, and `dropout` match training values.
- Works with `.npy` files for fast input/output handling.

---

## 📬 Contact

For questions or collaboration, feel free to create an issue or reach out!

---
