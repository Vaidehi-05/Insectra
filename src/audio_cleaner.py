import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, lfilter

TARGET_SR = 16000     # Sample rate
TARGET_DURATION = 4.0 # seconds (set same as training)

# Butterworth High-pass filter
def butter_highpass(cutoff, sr, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, sr, cutoff=1000):
    b, a = butter_highpass(cutoff, sr)
    return lfilter(b, a, data)

# Main Cleaning Pipeline
def clean_audio(input_path, output_path="cleaned.wav"):
    #  Load audio
    y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

    #  Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    #  High-pass filter (removes wind / low hum / rumble)
    y = highpass_filter(y, TARGET_SR)

    #  Noise reduction (spectral gating)
    y = nr.reduce_noise(y=y, sr=TARGET_SR)

    #  Normalize amplitude (volume leveling)
    y = librosa.util.normalize(y)

    #  Fix length to TARGET_DURATION seconds
    target_len = int(TARGET_DURATION * TARGET_SR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), 'constant')
    else:
        y = y[:target_len]

    #  Save cleaned audio
    sf.write(output_path, y, TARGET_SR)

    return output_path
