# 🎶 MelodyGAN: A Deep Learning Pipeline for Audio Feature Modeling 🎶

## 📁 Project Overview
This repository contains a complete audio processing pipeline, starting from data collection in `.wav` format to building two machine learning models:

1. **Autoencoder** – for dimensionality reduction and feature encoding
2. **GAN (Generative Adversarial Network)** – for generating new, high-quality audio features

These models are trained and used together to learn and synthesize audio representations effectively.

---

## 📅 Step 1: Data Collection (YouTube to WAV)

We use high-quality vocal and instrumental audio downloaded from YouTube using tools like `yt-dlp` or `youtube-dl`.

### Tools:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

### Example Command:
```bash
yt-dlp -x --audio-format wav <YouTube-URL>
```

The resulting `.wav` files are stored in a dataset directory for further processing.

---

## 🎛️ Step 2: Preprocessing with UVR5 (Ultimate Vocal Remover)

To separate vocals and instrumentals (or any stem-based separation), we use [UVR5](https://github.com/Anjok07/ultimatevocalremovergui).

### Output:
- `vocals.wav`
- `instrumental.wav`

These separated stems are used for specific feature learning or generation tasks (e.g., generating vocals only).

---

## 🎧 Step 3: Feature Extraction

We extract Mel Spectrogram and MFCC (Mel-Frequency Cepstral Coefficients) features using `torchaudio`.

### Features:
- `Mel Spectrogram`: Rich time-frequency representation of audio
- `MFCC`: Compressed perceptual features used in speech/audio processing

The features are averaged and concatenated to form a flattened feature vector, which is passed into the autoencoder.

---

## 🧠 Step 4: Model 1 - Autoencoder

### Purpose:
- Compresses high-dimensional audio features to a lower-dimensional latent space.
- Learns compact representations that retain essential info from original audio.

### Architecture:
- Fully connected feedforward neural network
- Input: 1D vector from feature extraction
- Hidden Layers: Reducing dimensions (e.g., 4096 → 2048 → 1024 → 512)
- Latent Layer: Final encoded representation
- Decoder mirrors encoder in reverse

### Output:
- Latent vector (compressed audio representation)

---

## 🤖 Step 5: Model 2 - MelodyGAN

### Purpose:
- Takes the latent representation from the autoencoder and learns to generate realistic feature vectors
- Helps synthesize new audio features

### Components:
- **Generator**: Learns to map random noise (or input conditions) to a plausible feature vector in latent space
- **Discriminator**: Distinguishes between real (autoencoded) and fake (generated) vectors

### Training Strategy:
- Generator and discriminator are trained adversarially
- Losses are minimized using backpropagation with adversarial loss functions

---

## 🔄 Integration Workflow

1. Extract features from audio using torchaudio
2. Encode the features using the trained autoencoder
3. Generate new latent representations using MelodyGAN
4. Decode generated features back to full-size feature vectors (optional)
5. Use for audio synthesis or evaluation

---

## 💡 Applications

- Music style transfer
- Vocal synthesis
- Data augmentation for audio datasets

---

## ⚖️ License
MIT License

---

## 🚀 Future Improvements
- Integration with diffusion models for waveform generation
- Incorporate temporal coherence in GAN
- Deploy as a web interface

---

Happy experimenting with MelodyGAN! 🎵🤝

