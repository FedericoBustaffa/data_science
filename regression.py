from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x_m = np.array([170, 175, 180, 185, 190]).reshape(-1, 1)
y_m = np.array([60, 70, 78, 87, 96]).reshape(-1, 1)

model = LinearRegression()
model.fit(x_m, y_m)

x = np.linspace(170, 190, 50).reshape(-1, 1)
y = model.predict(x)

print(f"M: {model.coef_}\tQ: {y[0][0] - model.coef_ * x[0][0]}")

plt.title("Regressione lineare")
plt.xlabel("Altezza")
plt.ylabel("Peso")
plt.grid(True)

plt.scatter(x_m, y_m, c="blue")
plt.plot(x, y, "b--")

x_f = np.array([150, 155, 160, 170, 180]).reshape(-1, 1)
y_f = np.array([40, 47, 50, 55, 65]).reshape(-1, 1)

model.fit(x_f, y_f)

x = np.linspace(150, 180, 50).reshape(-1, 1)
y = model.predict(x)

print(f"M: {model.coef_}\tQ: {y[0][0] - model.coef_ * x[0][0]}")

plt.scatter(x_f, y_f, c="red")
plt.plot(x, y, "r--")

plt.show()
