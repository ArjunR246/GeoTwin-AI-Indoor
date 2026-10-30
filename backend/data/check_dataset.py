import pandas as pd

df = pd.read_csv("data/processed/train.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())