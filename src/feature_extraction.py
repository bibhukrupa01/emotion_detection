import librosa
import numpy as np

def extract_features_from_audio(audio, sr):
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Extract Chroma feature
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    
    # Extract Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    
    return np.hstack((mfcc_mean, chroma_mean, mel_mean, contrast_mean))

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    return extract_features_from_audio(audio, sr)
