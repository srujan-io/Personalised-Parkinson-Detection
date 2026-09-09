import pandas as pd

file_path = "data/raw/parkinsons_data.csv"

df = pd.read_csv(file_path)



print("Shape:")
print(df.shape)

print("\nNumber of subjects:")
print(df["subject#"].nunique())



recordings = df.groupby("subject#").size()

print(recordings)


print(recordings.describe())



duration = df.groupby("subject#")["test_time"].agg(
    ["min", "max"]
)

duration["duration"] = duration["max"] - duration["min"]



print(duration)


print(duration["duration"].describe())




missing = df.isnull().sum()

print(missing[missing > 0])




voice_features = [
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE"
]



print(df[voice_features].describe().T)