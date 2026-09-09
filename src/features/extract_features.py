import librosa
import numpy as np


def extract_basic_features(audio, sample_rate):
    """
    Extract basic acoustic features from a preprocessed audio signal.
    """

    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    # MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    # Fundamental frequency / pitch
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    # Handle recordings where pitch cannot be detected
    if np.all(np.isnan(f0)):
        mean_f0 = 0.0
        std_f0 = 0.0
    else:
        mean_f0 = np.nanmean(f0)
        std_f0 = np.nanstd(f0)

    features = {
        "mean_rms": np.mean(rms),
        "std_rms": np.std(rms),

        "mean_zcr": np.mean(zcr),
        "std_zcr": np.std(zcr),

        "mean_f0": mean_f0,
        "std_f0": std_f0,

        "mfcc_mean": np.mean(mfcc, axis=1),
        "mfcc_std": np.std(mfcc, axis=1),
    }

    return features