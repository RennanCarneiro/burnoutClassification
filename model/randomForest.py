#%%
import pandas as pd

df = pd.read_csv("../data/processed/train_cleaned.csv")
df.head()
# %%
mapeamento_gender = {'Female':1, 'Male':0}
mapeamento_company = {'Service':1, 'Product':0}
mapeamento_WFH = {'Yes':1, 'No':0}

df['Gender'] = df['Gender'].map(mapeamento_gender)
df['Company Type'] = df['Company Type'].map(mapeamento_company)
df['WFH Setup Available'] = df['WFH Setup Available'].map(mapeamento_WFH)

bins = [0.0, 0.31, 0.45, 0.59, 1.0]
labels = [0,1,2,3]

df['Burn Rate'] = pd.cut(df['Burn Rate'], bins=bins, labels=labels, include_lowest=True)

target = 'Burn Rate'
features = ['Gender', 'Company Type', 'WFH Setup Available', 'Designation', 
              'Resource Allocation', 'Mental Fatigue Score']

X, y = df[features], df[target]
y
# %%
y.isnull().sum()
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# %%
from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(random_state=42,max_depth=15, n_estimators=300,)

modelo.fit(X_train, y_train)
# %%
y_pred = modelo.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# %%
%pip install shap

# %%
import shap
# %%
explicador = shap.TreeExplainer(modelo)

valores_shap = explicador.shap_values(X_test)

# %%
shap.summary_plot(valores_shap, X_test, plot_type="bar")
shap.summary_plot(valores_shap[:,:,3], X_test)
# %%
