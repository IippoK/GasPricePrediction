from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from data_parsing import get_data
import pandas as pd
import numpy as np

data = get_data()

# --- Shifting target one week forward ---
data["target_e95_next"] = data["e95_price"].shift(-1)
# data["target_diesel_next"] = data["diesel_price"].shift(-1)

data['e95_lag1'] = data['e95_price'].shift(1)


data["brent_change"] = data["brent_eur"] - data["brent_eur_lag1"]
data["e95_change"] = data["e95_price"] - data["e95_lag1"]
data["e95_current"] = data["e95_price"]

# --- Only past information ---
features = [
    "brent_eur",
    "brent_eur_lag1",
    "brent_eur_lag2",
    "usd_eur",
    "e95_lag1",
    "weeks_since_start",
    "season_sin",
    "season_cos"
]

features.append("brent_change")
features.append("e95_current")
features.append("e95_change")

data = data.dropna(subset=features + ["target_e95_next"]).reset_index(drop=True)

# Split features and targets
X = data[features]
y_e95 = data["target_e95_next"]
# y_diesel = data["target_diesel_next"]

# y = data["e95_price"]

# The chronological order of the data matters, so it shouldn't be shuffled
X_train, X_test, y_train, y_test = train_test_split(X, y_e95, test_size=0.2, shuffle=False)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Ridge Regression model training with regularization strength alpha
# Ridge over Linear to prevent overfitting of the model as Ridge penalizes too big coefs
model = Ridge(alpha=0.1)
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2:", r2)

# Feature names from polynomial expansion
feature_names = poly.get_feature_names_out(X.columns)

# Combine with coefficients
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_
})

# Sort by absolute magnitude (strongest influence first)
coef_df['abs_coef'] = np.abs(coef_df["coefficient"])
coef_df = coef_df.sort_values('abs_coef', ascending=False)

# Display top 15
print('\nTop 15 most influential features:')
print(coef_df.head(15))

# print(poly1.get_feature_names_out(X_train.columns))

# Naïve baseline: next week's price = this week's price
y_naive = y_test.shift(1)  # shift forward by one week

# Compute metrics for comparison
mae_naive = mean_absolute_error(y_test[1:], y_naive[1:])
r2_naive = r2_score(y_test[1:], y_naive[1:])

print("\nNaïve baseline:")
print(f"MAE: {mae_naive:.3f}")
print(f"R2: {r2_naive:.3f}")

# Compare your model improvement
improvement = (mae_naive - mae) / mae_naive * 100
print(f"Model improves MAE over naïve by {improvement:.1f}%")
