# Insectra: Insect Acoustic Detection & Classification 🦗🎧

## A real-time bioacoustic insect detection system built for agriculture & ecology 🌾🔊

>Visual insect detection fails in darkness, dense crops, and camouflage-heavy environments.
This project proves that sound alone is enough.

Insect Acoustic Detection & Classification is a machine-learning–powered system that identifies insect species using wingbeat frequencies, harmonic patterns, and MFCC-based acoustic features. Designed for CPU-only deployment, the system achieves high accuracy while remaining lightweight and field-ready.

## About📝⭐<br>

Insects play a critical role in agriculture, ecology, and public health — but they are also responsible for massive crop losses every year. Traditional monitoring methods rely on manual inspection or image-based models that break down in real-world conditions.
This project explores acoustic insect recognition as a robust alternative. Instead of images, it analyzes bioacoustic signatures produced by insect wingbeats and stridulation, enabling detection even in:
✅ Low-light environments 🌙
✅ Dense vegetation 🌿
✅ Nocturnal settings 🕷️
✅ Camouflage-heavy conditions 🟢

The system was developed as an end-to-end ML pipeline — from dataset creation and feature engineering to model evaluation and deployment — with a strong emphasis on practicality and real-world usability with an ultimate aim of actual field deployment.

## Features 📖✨

🎧 Purely Acoustic Detection (No Images Needed)
✔️ Works in darkness, clutter, and visually occluded environments

🧠 266-Dimensional Engineered Feature Vector
✔️ MFCC means & standard deviations
✔️ Delta & delta-delta MFCCs
✔️ Spectral descriptors (centroid, rolloff, bandwidth)
✔️ Harmonic wingbeat peak frequencies

📊 Multi-Class Classification
✔️ Chorthippus biguttulus (Grasshopper)
✔️ Gryllus bimaculatus (Field cricket)
✔️ Ruspolia nitidula (Katydid)
✔️ Other insects
✔️ Environmental / No-insect noise

🚀 XGBoost-Based Final Model
✔️ Selected after benchmarking 10+ ML & DL models
✔️ High accuracy with low inference latency
✔️ Strong performance on minority classes

⚙️ CPU-Only, Real-Time Friendly
✔️ 15–22 ms inference time
✔️ Suitable for edge & field deployment

🧪 Noise-Robust Preprocessing
✔️ Noise trimming
✔️ RobustScaler to preserve biological outliers

## Preview 👀
![WhatsApp Image 2025-12-19 at 8 19 53 AM (1)](https://github.com/user-attachments/assets/c74e47a9-1211-494d-a579-eecd7392fe3d)
![WhatsApp Image 2025-12-19 at 8 19 53 AM](https://github.com/user-attachments/assets/36b65313-8e0c-4b88-a993-d2a4aa9be43b)
![WhatsApp Image 2025-12-19 at 8 19 52 AM](https://github.com/user-attachments/assets/fc8d0702-89d5-4815-b6fc-ed3b14e9619c)
📌 Sample Insect Audio Files
Included .wav files(/TrialAudio) allow users to test the model immediately without external datasets.
> Note: Sample audio files are intentionally included for model testing and reproducibility.

## Tech Stack 🛠️🥇
| Technology     | Purpose                               |
| -------------- | ------------------------------------- |
| Python         | Core implementation                   |
| Librosa        | Audio processing & feature extraction |
| NumPy / Pandas | Numerical & data handling             |
| Scikit-learn   | ML utilities & preprocessing          |
| XGBoost        | Final classification model            |
| PyTorch        | 1D-CNN experimentation                |
| Flask          | Web-based inference interface         |
| HTML / CSS     | Frontend UI                           |

## How It Works 🛠️

1️⃣ Raw insect audio (.wav) is provided
2️⃣ Noise trimming & signal normalization
3️⃣ Extraction of 266 acoustic features
4️⃣ Features scaled using RobustScaler
5️⃣ XGBoost model predicts insect class
6️⃣ Result displayed with insect details

This pipeline allows fast, accurate, and interpretable predictions using only audio signals.

## Dataset Overview 📊
| Class                  | Samples |
| ---------------------- | ------- |
| Chorthippus biguttulus | 1016    |
| Gryllus bimaculatus    | 587     |
| Ruspolia nitidula      | 366     |
| Other Insects          | 3000    |
| Environment / Noise    | 2500    |

Data Sources:
Xeno-Canto · Macaulay Library · EcoSounds · InsectSound1000 · ESC-50 · Zenodo

> Both controlled and field recordings were used to ensure realistic variability.

## Model Evaluation 🧪
Multiple models were trained and evaluated:
- SVM (RBF)
- Random Forest
- Logistic Regression
- LightGBM
- CatBoost
- Extra Trees
- AdaBoost
- HistGradientBoosting
- 1D CNN
- XGBoost (Final)

#### Why XGBoost?
✔️ Best class-wise F1 scores
✔️ Stable probability calibration
✔️ Robust under noisy conditions
✔️ Lightweight and CPU-efficient

## Getting Started ⚡
### Prerequisites 📌

✅ Python 3.10+
✅ pip
✅ Any OS (Windows / macOS / Linux)

### Setup & Run 🚀
```
# Clone the repository
git clone https://github.com/Vaidehi-05/Insectra.git

# Navigate to project directory
cd Insectra

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python flask_app/app.py
```
Visit:
📍 http://127.0.0.1:5000/

## Testing & Validation 🧪⚠️

📌 Unit testing for feature extraction & scaling
📌 Integration testing for preprocessing → model
📌 End-to-end system testing via Flask
📌 Robustness testing with noisy and clipped audio
📌 Performance testing under batch inference
📌 Average inference time: 15–22 ms (CPU)

## Future Enhancements 🌱

🚀 Expand dataset (more species, seasons, geographies)
🚀 Transformer-based audio models (AST, WaveNet)
🚀 Continuous audio stream monitoring
🚀 ONNX / Treelite optimization for edge devices
🚀 Environmental metadata fusion
🚀 Farmer-friendly mobile/web dashboard

> Happy Coding & Happy Research! 🦗🎧🥇

