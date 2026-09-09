import pandas as pd


def load_telemonitoring_data(file_path):
    """
    Load the UCI Parkinson's Telemonitoring dataset.
    """

    df = pd.read_csv(file_path)

    return df


if __name__ == "__main__":

    file_path = "data/raw/parkinsons_data.csv"

    df = load_telemonitoring_data(file_path)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nNumber of unique subjects:")
    print(df["subject#"].nunique())

    print("\nRecordings per subject:")
    print(df["subject#"].value_counts().head(10))