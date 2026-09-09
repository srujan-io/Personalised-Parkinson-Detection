import librosa
import numpy as np


def load_audio(file_path, target_sr=16000):

    audio, sr = librosa.load(
        file_path,
        sr=target_sr,
        mono=True
    )

    return audio, sr


def trim_silence(audio, top_db=30):

    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=top_db
    )

    return trimmed_audio


def normalize_audio(audio):
    """
    Normalize audio amplitude to the range [-1, 1].
    """

    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    return audio / max_amplitude


def preprocess_audio(
    file_path,
    target_sr=16000,
    top_db=30
):
    """
    Complete preprocessing pipeline.
    """

    # Load audio
    audio, sr = load_audio(
        file_path,
        target_sr
    )

    # Remove silence
    audio = trim_silence(
        audio,
        top_db
    )

    # Normalize amplitude
    audio = normalize_audio(audio)

    return audio, sr