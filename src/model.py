from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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

# --- Only past information ---
features = [
    "brent_eur_lag1",
    "brent_eur_lag2",
    "usd_eur",
    "e95_lag1",
    "weeks_since_start",
    "season_sin",
    "season_cos"
]

features.append("brent_change")
features.append('e95_lag1')

"""
features = ["brent_eur", "brent_eur_lag1", "brent_eur_lag2", "weeks_since_start", "season_sin", "season_cos"]
"""

data = data.dropna(subset=features + ["target_e95_next"]).reset_index(drop=True)

# Split features and targets
X = data[features]
y_e95 = data["target_e95_next"]
# y_diesel = data["target_diesel_next"]

# y = data["e95_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y_e95, test_size=0.2, shuffle=False)
print('Feature columns:', list(X.columns))
print('X_train shape before poly:', X_train.shape)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print('X_train_poly shape after poly:', X_train.shape)
print('X_test_poly shape after poly:', X_test_poly.shape)


# Ridge Regression model training with regularization strength alpha
model = Ridge(alpha=0.5)
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

"""
model = LinearRegression()

# model.fit(X_train_poly, y_train)
# model.fit(X_train, y_train)


# --- Random Forest Regressor ---
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest results:")
print(f"MAE: {rf_mae:.3f}")
print(f"R2: {rf_r2:.3f}")


print("Intercept:", model.intercept_)
print(f"{features[0]} coefficient:", model.coef_[0])
print(f"{features[1]} coefficient:", model.coef_[1])
print(f"{features[2]} coefficient:", model.coef_[2])
print(f"{features[3]} coefficient:", model.coef_[3])
print(f"{features[4]} coefficient:", model.coef_[4])
# print(f"{features[5]} coefficient:", model.coef_[5])
print("coefficients:", model.coef_)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

print("\nRandom Forest results:")
print(f"MAE: {rf_mae:.3f}")
print(f"R2: {rf_r2:.3f}")

df = get_data()
df[['euro95', 'brent_usd', 'rate']].corr()
print(df)
"""

print("MAE:", mae)
print("R2:", r2)



"""
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
print(coef_df)
"""
# print(poly1.get_feature_names_out(X_train.columns))
