import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("final_dataset.csv")

df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

x = df.drop('AQI', axis=1)
y = df['AQI']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

joblib.dump(model, 'model.joblib')
joblib.dump(list(x.columns), 'features.joblib') 