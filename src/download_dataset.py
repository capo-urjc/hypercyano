import pandas as pd

URL = "https://zenodo.org/records/14056022/files/cyanobacterial_spectra_processed.xlsx?download=1"

df = pd.read_excel(URL)
print(f"Original file: {len(df)} rows")

df_clean = df.drop_duplicates()
print(f"Clean file: {len(df_clean)} rows")
print(f"Removed {len(df) - len(df_clean)} duplicate rows")

df_clean.to_excel("../data/cyanobacterial_spectra_processed_clean.xlsx", index=False)
print("File saved without duplicates!")