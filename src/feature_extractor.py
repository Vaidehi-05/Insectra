import numpy as np
import librosa
import scipy.stats
from scipy.signal import find_peaks

TARGET_SR = 16000
N_MFCC = 40   # <<-- set this to the n_mfcc you used in training

def band_energy_ratios(S, sr):
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(S)

    bands = [
        (0, 500),       # low
        (500, 2000),    # mid-low
        (2000, 6000),   # mid-high
        (6000, 8000)    # high
    ]

    ratios = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        ratios.append(np.sum(S[mask]) / total_energy if total_energy > 0 else 0)
    return ratios  # 4 values


def top_peak_frequencies(y, sr):
    S = np.abs(librosa.stft(y))
    spectrum = np.mean(S, axis=1)
    peaks, _ = find_peaks(spectrum, distance=20)
    peak_freqs = librosa.fft_frequencies(sr=sr)[peaks]

    peak_freqs_sorted = sorted(peak_freqs[:3]) if len(peak_freqs) >= 3 else [0,0,0]
    return peak_freqs_sorted[:3]


def spectral_entropy(y):
    S = np.abs(librosa.stft(y))
    ps = S / np.sum(S)
    entropy = -np.sum(ps * np.log2(ps + 1e-12))
    return entropy


def crest_factor(y):
    return np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-12)


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Delta + delta2
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    spectral_centroid_mean = np.mean(centroid)
    spectral_centroid_std = np.std(centroid)
    spectral_bandwidth_mean = np.mean(bandwidth)
    spectral_bandwidth_std = np.std(bandwidth)
    spectral_contrast_mean = np.mean(contrast)
    spectral_contrast_std = np.std(contrast)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    flatness_mean = np.mean(flatness)
    flatness_std = np.std(flatness)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral entropy & crest factor
    entropy_val = spectral_entropy(y)
    crest = crest_factor(y)

    # Peak frequencies
    peak1, peak2, peak3 = top_peak_frequencies(y, sr)

    # Band energy ratios
    S = np.abs(librosa.stft(y))
    low, midlow, midhigh, high = band_energy_ratios(S, sr)

    # Onset rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = np.mean(onset_env)

    # Signal-to-noise ratio
    snr = rms_mean / (rms_std + 1e-12)

    # ---- FINAL FEATURE VECTOR ----
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std,
        [
            spectral_centroid_mean, spectral_centroid_std,
            spectral_bandwidth_mean, spectral_bandwidth_std,
            spectral_contrast_mean, spectral_contrast_std,
            rolloff_mean, rolloff_std,
            flatness_mean, flatness_std,
            zcr_mean, zcr_std,
            rms_mean, rms_std,
            entropy_val, crest,
            low, midlow, midhigh, high,
            peak1, peak2, peak3,
            onset_rate, snr
        ]
    ])

    return features
