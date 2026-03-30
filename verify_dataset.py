import pandas as pd

df = pd.read_csv("data/winequality-red.csv", sep=';')
print(df.head())
print(df.shape)
print(df.columns)

df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)