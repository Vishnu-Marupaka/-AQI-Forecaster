import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/vishn/Downloads/final_dataset.csv")


df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True)) 

print(f"After cleaning no of rows in data: {len(df)}")

x = df.drop('AQI', axis=1) 
y = df['AQI']

print(f"Training with {x.shape[1]} features: {list(x.columns)}")


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Square Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Model Accuracy: Actual vs. Predicted AQI")
plt.show()

