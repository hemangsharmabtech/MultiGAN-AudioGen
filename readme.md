---

# 🎶 MelodyGAN: A VQ-VAE + GAN-based Audio Generation Pipeline

Welcome to **MelodyGAN**, an end-to-end audio processing and generation pipeline combining the power of **Vector Quantized Variational Autoencoders (VQ-VAE)** and **Generative Adversarial Networks (GANs)**. From real-world audio sourcing to intelligent waveform generation, this project showcases the full cycle of audio synthesis powered by deep learning and modern audio preprocessing techniques.

---

## 🧰 Tools & Technologies Used

- Open-source audio downloaders (e.g., YouTube-DL, SpotDL, etc.)
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) for vocal/instrument separation
- Python & PyTorch
- Librosa for audio preprocessing
- Optuna for hyperparameter tuning
- Custom VQ-VAE architecture
- GANs for latent space modeling

---

## 🎧 Step 1: Data Collection

High-quality `.wav` audio files were sourced using various open-source tools available across the internet (such as YouTube downloaders). This ensured a wide range of musical styles and vocal types for robust training.

---

## 🛠️ Step 2: Preprocessing

All audio files underwent consistent and clean processing:

1. **Format Conversion**:  
   - Converted to Mono channel  
   - Resampled to 22.5kHz  
   - Quantized to 16-bit depth  

2. **Vocal/Instrument Separation**:  
   - Utilized **UVR5** to separate vocal and instrumental stems.

3. **Manual Cleaning**:  
   - Removed silent sections  
   - Normalized audio levels  
   - Trimmed unnecessary background noise manually

4. **Feature Extraction**:
   - Converted audio into **Mel spectrograms**
   - Extracted **MFCCs** for frequency-based analysis

---

## 🧠 Step 3: Model Architecture

### 🧩 VQ-VAE (Vector Quantized Variational Autoencoder)

- **Encoder**:  
  Converts spectrogram input into a discrete latent space using vector quantization.

- **Codebook**:  
  Acts as a learned dictionary to represent compressed features.

- **Decoder**:  
  Reconstructs the original spectrogram from quantized latent vectors.

This architecture enables efficient representation learning of complex audio features.

### 🧪 Optimization
- **Optuna** was used for hyperparameter tuning (e.g., learning rates, codebook sizes, number of layers).

---

## 🤖 Step 4: GAN on Latent Space

Once the VQ-VAE was trained, the discrete latent vectors were used as inputs to train a **Generative Adversarial Network**.

- **Generator**:  
  Learns to generate realistic latent representations mimicking VQ-VAE’s latent space.

- **Discriminator**:  
  Differentiates between real encoded latent vectors and generated ones.

This GAN helps synthesize new audio content from scratch by decoding generated latent vectors via the VQ-VAE decoder.

---

## 🔁 How It All Comes Together

```mermaid
graph LR
    A[Raw Audio (WAV)] --> B[Preprocessing (Mono, Trim, Normalize)]
    B --> C[UVR5 Separation]
    C --> D[Mel & MFCC Extraction]
    D --> E[Encoder (VQ-VAE)]
    E --> F[Latent Codebook]
    F --> G[GAN Generator]
    G --> H[Latent Vectors]
    H --> I[Decoder (VQ-VAE)]
    I --> J[Reconstructed Audio]
```

---

## 🚀 Future Work

- Expand dataset diversity for broader genre coverage
- Explore diffusion-based generation methods
- Real-time generation via lightweight models

---

## 📝 License

This project is open-sourced for educational and experimental use. Please ensure proper credit when using components.

---
