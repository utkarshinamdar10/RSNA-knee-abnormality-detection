import pandas as pd
import os

data_dir = "data"

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        print(f"=== {filename} ===")
        try:
            df = pd.read_csv(filepath)
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print("Head:")
            print(df.head(2))
            print("Missing values:")
            print(df.isnull().sum())
            print("\n")
        except Exception as e:
            print(f"Error reading {filename}: {e}\n")
