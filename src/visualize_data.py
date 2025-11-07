import matplotlib.pyplot as plt
from src.data_parsing import get_data
import src.model as variables

test_dates = variables.data.iloc[variables.y_test.index]['date']

"""
plt.figure()
plt.scatter(get_data().merged["brent_price"], get_data().merged["e95_price"])
plt.xlabel("Brent Oil Price")
plt.ylabel("E95 Price")

plt.figure()
plt.scatter(get_data().merged["usd_eur"], get_data().merged["e95_price"])
plt.xlabel("USD/EUR")
plt.ylabel("E95 Price")
"""


plt.figure(figsize=(12, 6))
plt.plot(variables.y_test.index, variables.y_test, label="actual E95 Price", linewidth=2)
plt.plot(variables.y_test.index, variables.y_pred, label="Predicted E95 Price", linewidth=2, linestyle="--")

plt.title("Predicted vs Actual E95 Fuel Price (Weekly)")
plt.xlabel("Time (Weeks)")
plt.ylabel("Price (€/L)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""
plt.figure(figsize=(12, 6))
plt.plot(test_dates, variables.y_test, label="Actual E95", linewidth=2)
plt.plot(test_dates, variables.y_pred, label="Ridge Prediction", linestyle="--")
plt.plot(test_dates, variables.rf_pred, label="Random Forest Prediction", linestyle=":")
plt.title("Model Comparison: Ridge vs Random Forest")
plt.xlabel("Date")
plt.ylabel("Price (€ per liter)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

"""