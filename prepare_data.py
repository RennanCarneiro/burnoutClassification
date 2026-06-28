# %%
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df.head()
df.info()
df.isnull().sum()

# %%
df = df.dropna(subset=['Burn Rate'])
df['Resource Allocation'] = df['Resource Allocation'].fillna(df['Resource Allocation'].median())
df['Mental Fatigue Score'] = df['Mental Fatigue Score'].fillna(df['Mental Fatigue Score'].median())

df.isnull().sum() #tudo preenchido com a mediana
df.info() #verificar quantas linhas sobraram
# %%
df.describe()
# %%
df.to_csv("data/processed/train_cleaned.csv", index=False)

# %%
df2 = pd.read_csv("data/processed/train_cleaned.csv")
df2.isnull().sum() #conferindo que ta tudo certo
