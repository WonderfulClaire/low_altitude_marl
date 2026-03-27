from pathlib import Path
import pandas as pd

root = Path("outputs")
csv_files = list(root.rglob("*.csv"))

print(f"Found {len(csv_files)} csv files")
for f in csv_files[:10]:
    print(f)

if csv_files:
    df = pd.read_csv(csv_files[0])
    print(df.head())
    print(df.columns.tolist())
