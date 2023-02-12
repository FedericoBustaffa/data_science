from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data_set.csv")

x_m = np.array([150, 155, 160, 165, 170, 175, 180, 185, 190]).reshape(-1, 1)
y_m = np.array([60, 65, 70, 75, 80, 85, 90, 95, 100]).reshape(-1, 1)

model = LinearRegression()
model.fit(x_m, y_m)

x = np.linspace(150, 190, 50).reshape(-1, 1)
y = model.predict(x)

print(f"M: {model.coef_}\tQ: {y[0][0] - model.coef_ * x[0][0]}")

plt.title("Regressione lineare")
plt.xlabel("Altezza")
plt.ylabel("Peso")
plt.grid(True)

plt.scatter(x_m, y_m, c="blue")
plt.plot(x, y, "b--")

x_f = np.array([150, 155, 160, 165, 170, 175, 180, 185, 190]).reshape(-1, 1)
y_f = np.array([50, 55, 60, 65, 75, 80, 85, 90, 95]).reshape(-1, 1)

model.fit(x_f, y_f)
y = model.predict(x)

print(f"M: {model.coef_}\tQ: {y[0][0] - model.coef_ * x[0][0]}")

plt.scatter(x_f, y_f, c="red")
plt.plot(x, y, "r--")

plt.show()
