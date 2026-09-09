import numpy as np
import soundfile as sf

from src.data.preprocessing import preprocess_audio
from src.features.extract_features import extract_basic_features




sample_rate = 16000
duration = 5

t = np.linspace(
    0,
    duration,
    int(sample_rate * duration),
    endpoint=False
)

# Simulated voiced signal
frequency = 150
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Add a small amount of noise ( little)
noise = 0.02 * np.random.randn(len(t))

audio = audio + noise


# Save test audio
test_file = "experiments/test_audio.wav"

sf.write(
    test_file,
    audio,
    sample_rate
)

print("Test audio created.")


# --------------------------------------------------
# 2. Preprocess audio
# --------------------------------------------------

processed_audio, sr = preprocess_audio(test_file)

print(f"Sampling rate: {sr}")
print(f"Number of samples: {len(processed_audio)}")
print(f"Duration: {len(processed_audio) / sr:.2f} seconds")


# --------------------------------------------------
# 3. Extract features
# --------------------------------------------------

features = extract_basic_features(
    processed_audio,
    sr
)


# --------------------------------------------------
# 4. Display features
# --------------------------------------------------

print("\nExtracted Features:")
print("-------------------")

for name, value in features.items():

    print(f"{name}: {value}")