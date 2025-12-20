# Insectra: Insect Acoustic Detection & Classification 🦗🎧
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)<br>
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)<br>
## A lightweight, real-time bioacoustic system for detecting and classifying insect species using engineered audio features and machine learning. 🍀
<br>

>Visual insect detection fails in darkness, dense crops, and camouflage-heavy environments.
This project proves that sound alone is enough.

Insect Acoustic Detection & Classification is a machine-learning–powered system that identifies insect species using wingbeat frequencies, harmonic patterns, and MFCC-based acoustic features. Designed for CPU-only deployment, the system achieves high accuracy while remaining lightweight and field-ready.
<br>
## About📝⭐

Insects play a critical role in agriculture, ecology, and public health — yet they are also responsible for up to 40% global crop loss annually. Traditional insect monitoring systems rely heavily on manual inspection or image-based models, both of which fail under real-world conditions such as low light, occlusion, camouflage, and nocturnal activity. <br><br>
This project explores acoustic insect recognition as a robust alternative. Instead of images, it analyzes bioacoustic signatures produced by insect wingbeats and stridulation, enabling detection even in:<br>
✅ Low-light environments 🌙<br>
✅ Dense vegetation 🌿<br>
✅ Nocturnal settings 🕷️<br>
✅ Camouflage-heavy conditions 🟢<br>
<br>
The system was developed as an end-to-end ML pipeline — from dataset creation and feature engineering to model evaluation and deployment — with a strong emphasis on practicality, real-world usability and with an ultimate aim of actual field deployment.<br>

## Features 📖✨

🎧 Purely Acoustic Detection (No Images Needed)<br>
✔️ Works in darkness, clutter, and visually occluded environments<br>
<br>
🧠 266-Dimensional Engineered Feature Vector<br>
✔️ MFCC means & standard deviations<br>
✔️ Delta & delta-delta MFCCs<br>
✔️ Spectral descriptors (centroid, rolloff, bandwidth)<br>
✔️ Harmonic wingbeat peak frequencies<br>
<br>
📊 Multi-Class Classification<br>
✔️ Chorthippus biguttulus (Grasshopper)<br>
✔️ Gryllus bimaculatus (Field cricket)<br>
✔️ Ruspolia nitidula (Katydid)<br>
✔️ Other insects<br>
✔️ Environmental / No-insect noise<br>
<br>
🚀 XGBoost-Based Final Model<br>
✔️ Selected after benchmarking 10+ ML & DL models<br>
✔️ High accuracy with low inference latency<br>
✔️ Strong performance on minority classes<br>
<br>
🎙️ Flexible Audio Input (Upload or Record)<br>
✔️ Users can upload pre-recorded .wav files<br>
✔️ Live audio recording supported directly through the interface<br>
✔️ Enables instant testing without external audio files<br>
<br>
⚙️ CPU-Only, Real-Time Friendly<br>
✔️ 15–22 ms inference time<br>
✔️ Suitable for edge & field deployment<br>
<br>
🧪 Noise-Robust Preprocessing<br>
✔️ Noise trimming<br>
✔️ RobustScaler to preserve biological outliers<br>
<br>
## Target Classes 🐞
| Class                       | Description                           |
| --------------------------- | ------------------------------------- |
| **Chorthippus biguttulus**  | Grasshopper (major crop pest)         |
| **Gryllus bimaculatus**     | Field cricket (nocturnal pest)        |
| **Ruspolia nitidula**       | Katydid (high-frequency foliage pest) |
| **Other Insects**           | Non-target insect sounds              |
| **Environment / No Insect** | Ambient background noise              |

## Preview 👀
#### 📌 Opening Screen<br>
![WhatsApp Image 2025-12-19 at 8 19 52 AM](https://github.com/user-attachments/assets/fc8d0702-89d5-4815-b6fc-ed3b14e9619c)
<br>
#### 📌 Testing page<br>
![WhatsApp Image 2025-12-19 at 8 19 53 AM (1)](https://github.com/user-attachments/assets/c74e47a9-1211-494d-a579-eecd7392fe3d)
<br>
#### 📌 Result Page <br>
![WhatsApp Image 2025-12-19 at 8 19 53 AM](https://github.com/user-attachments/assets/36b65313-8e0c-4b88-a993-d2a4aa9be43b)

<br>

#### 📌 Sample Insect Audio Files<br>
Included .wav files(/TrialAudio) allow users to test the model immediately without external datasets.<br>

>Note: Sample audio files are intentionally included for model testing and reproducibility.<br>

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

1️⃣ Raw insect audio (.wav) is provided<br>
2️⃣ Noise trimming & signal normalization<br>
3️⃣ Extraction of 266 acoustic features<br>
4️⃣ Features scaled using RobustScaler<br>
5️⃣ XGBoost model predicts insect class<br>
6️⃣ Result displayed with insect details<br>
<br>
This pipeline allows fast, accurate, and interpretable predictions using only audio signals.<br>

## Dataset Overview 📊
| Class                  | Samples |
| ---------------------- | ------- |
| Chorthippus biguttulus | 1016    |
| Gryllus bimaculatus    | 587     |
| Ruspolia nitidula      | 366     |
| Other Insects          | 3000    |
| Environment / Noise    | 2500    |

Data Sources:<br>
Xeno-Canto · Macaulay Library · EcoSounds · InsectSound1000 · ESC-50 · Zenodo<br>

> Both controlled and field recordings were used to ensure realistic variability.

## Model Evaluation 🧪
Multiple models were trained and evaluated:<br>
- SVM (RBF)<br>
- Random Forest<br>
- Logistic Regression<br>
- LightGBM<br>
- CatBoost<br>
- Extra Trees<br>
- AdaBoost<br>
- HistGradientBoosting<br>
- 1D CNN<br>
- XGBoost (Final)<br>

#### Why XGBoost?
✔️ Best class-wise F1 scores<br>
✔️ Stable probability calibration<br>
✔️ Robust under noisy conditions<br>
✔️ Lightweight and CPU-efficient<br>
<hr>

## Getting Started ⚡
### Prerequisites 📌

✅ Python 3.10+<br>
✅ pip<br>
✅ Any OS (Windows / macOS / Linux)<br>

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
<hr>

## Testing & Validation 🧪⚠️

📌 Unit testing for feature extraction & scaling<br>
📌 Integration testing for preprocessing → model<br>
📌 End-to-end system testing via Flask<br>
📌 Robustness testing with noisy and clipped audio<br>
📌 Performance testing under batch inference<br>
📌 Average inference time: 15–22 ms (CPU)<br>

## Future Enhancements 🌱

🚀 Expand dataset (more species, seasons, geographies)<br>
🚀 Transformer-based audio models (AST, WaveNet)<br>
🚀 Continuous audio stream monitoring<br>
🚀 ONNX / Treelite optimization for edge devices<br>
🚀 Environmental metadata fusion<br>
🚀 Farmer-friendly mobile/web dashboard<br>

> Happy Coding & Happy Research! 🦗🎧🥇

