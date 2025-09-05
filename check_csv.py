import pandas as pd

df = pd.read_csv("sms-spam.csv", encoding="latin-1")
print("👉 Columns in CSV:", df.columns)
print("👉 First 5 rows:")
print(df.head())