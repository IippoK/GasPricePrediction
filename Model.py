from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from DataParsing import get_data

data = get_data()

# --- Shifting target one week forward ---
data["target_e95_next"] = data["e95_price"].shift(-1)
# data["target_diesel_next"] = data["diesel_price"].shift(-1)

# --- Only past information ---
features = [
    "brent_eur_lag1",
    "brent_eur_lag2",
    "weeks_since_start",
    "season_sin",
    "season_cos"
]

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

model = LinearRegression()
model.fit(X_train, y_train)

# --- Random Forest Regressor ---
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

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
