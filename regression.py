"""Regression example
Basic linear regression sample
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data set
data = pd.read_csv("data.csv")
heights = np.array(data.altezza).reshape(-1, 1)
weights = np.array(data.peso).reshape(-1, 1)

# learning
model = LinearRegression()
model.fit(heights, weights)

# samples
x = np.linspace(100, 250, 50).reshape(-1, 1)
y = model.predict(x)

# coefficienti retta regressione
print(f"coefficiente angolare: {model.coef_}")
print(f"intersezione ordinate: {y[0][0] - model.coef_ * x[0][0]}")

# plotting
plt.title("Regressione lineare")
plt.xlabel("Altezza")
plt.ylabel("Peso")
plt.xlim(145, 205)
plt.ylim(0, 150)
plt.grid(True)

plt.scatter(heights, weights, color="black")
plt.plot(x, y, "r", label="reg_line")

plt.show()
