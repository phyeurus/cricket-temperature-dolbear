import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import scipy.stats as stats
import joblib


raw_data = [
    [19, 88.59999847], [16, 71.59999847], [22, 93.30000305], [17, 84.30000305],
    [19, 80.59999847], [19, 75.19999695], [17, 6969999695], [18, 82], [15, 69.4000153],
    [18, 83.30000305], [16, 79.59999847], [16, 82.59999847], [19, 80.59999847],
    [17, 83.5], [17, 76.30000305], [113, 62], [15, 65], [16, 76], [16, 81], [16, 76],
    [16, 81], [16, 66], [17, 74], [16, 68], [19, 75], [18, 84], [17, 86], [20, 80],
    [21, 89], [18, 90], [18, 81], [13, 62], [16, 83], [17, 85], [14, 63], [18, 70],
    [16, 74], [21, 85], [17, 74], [15, 69], [16, 78], [16, 73], [19, 85], [19, 83],
    [16, 74], [16, 81], [16, 66], [21, 89], [15, 68], [18, 88], [21, 90], [19, 78],
    [15, 65], [18, 83], [16, 65], [19, 88], [21, 85], [20, 88], [17, 82], [17, 76],
    [14, 64], [17, 78], [19, 81], [15, 62], [16, 79], [16, 72], [16, 69], [16, 70],
    [18, 77], [16, 67], [16, 68], [17, 73], [16, 83], [20, 84], [18, 74], [17, 86],
    [16, 74], [16, 69], [18, 79], [19, 78], [15, 63], [14, 63], [19, 77], [19, 78],
    [14, 63], [17, 69], [20, 87], [19, 76], [17, 71], [17, 71], [20, 82], [16, 68],
    [19, 86], [16, 66], [17, 82], [20, 87], [17, 77], [15, 69], [17, 75], [16, 78],
    [17, 67], [15, 63], [17, 67], [20, 88], [18, 84], [19, 90], [15, 65], [19, 85],
    [19, 88], [18, 89], [18, 84], [18, 80], [16, 84], [16, 63], [18, 72], [18, 87],
    [16, 69], [16, 78], [19, 87], [15, 64], [16, 68], [16, 72], [19, 85], [20, 83],
    [15, 68], [16, 69], [16, 79], [19, 85], [16, 67], [17, 74], [16, 82], [18, 77],
    [16, 63], [16, 81], [16, 79], [18, 71], [15, 65], [17, 73], [19, 82], [16, 66],
    [17, 67], [19, 87], [15, 62], [16, 71], [16, 64], [15, 69], [18, 72], [18, 81],
    [19, 75], [19, 86], [17, 82], [15, 65], [16, 83], [15, 66], [18, 81], [18, 77]
]

df = pd.DataFrame(raw_data, columns=['temperature_c', 'chirps_per_minute'])

df = df[(df['temperature_c'] >= 0) & (df['temperature_c'] <= 50)]
df = df[df['chirps_per_minute'] < 200]

print(f"Clean data points: {len(df)}")
print(df.head())

X = df[['chirps_per_minute']]
y = df['temperature_c']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = LinearRegression()
model.fit(X_train, y_train)

intercept = model.intercept_
slope = model.coef_[0]

print(f"\nLinear regression equation:")
print(f"Temperature = {intercept:.4f} + {slope:.4f} * chirps_per_minute")


y_test_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"\nTest set evaluation:")
print(f"R² = {r2_test:.4f}")
print(f"MAE = {mae_test:.2f} deg C")


 
 
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n5-fold cross-validation R²: mean = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
 
 
 
y_full_pred = model.predict(X)
residuals = y - y_full_pred

plt.figure(figsize=(8,4))
plt.scatter(y_full_pred, residuals, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted temperature (deg C)')
plt.ylabel('Residuals (deg C)')
plt.title('Residual plot (linear model check)')
plt.grid(True, alpha=0.3)
plt.show()

 

def prediction_interval(model, X_train, y_train, x0, confidence=0.95):
    """
    Returns (prediction, lower_bound, upper_bound) for a single x0.
    """
    n = len(X_train)
    dof = n - 2
    t_val = stats.t.ppf((1 + confidence) / 2, dof)
    
    y_pred = model.predict([[x0]])[0]
    y_train_pred = model.predict(X_train)
    mse = np.mean((y_train - y_train_pred) ** 2)
    
    x_mean = X_train.mean().values[0]
    Sxx = np.sum((X_train.values.flatten() - x_mean) ** 2)
    
    se = np.sqrt(mse * (1 + 1/n + ((x0 - x_mean) ** 2) / Sxx))
    margin = t_val * se
    
    return y_pred, y_pred - margin, y_pred + margin

 
x_new = 130
pred, lower, upper = prediction_interval(model, X_train, y_train, x_new)
print(f"\nPrediction for {x_new} chirps/min:")
print(f"  Point estimate: {pred:.2f} deg C")
print(f"  95% prediction interval: [{lower:.2f}, {upper:.2f}] deg C")

 

def predict_temp_from_chirps(chirps, time_unit='minute'):
    """
    chirps : number of chirps counted
    time_unit : 'second', '15sec', or 'minute' (default)
    Returns predicted temperature in deg C.
    """
    if time_unit == 'second':
        chirps_per_min = chirps * 60
    elif time_unit == '15sec':
        chirps_per_min = chirps * 4
    elif time_unit == 'minute':
        chirps_per_min = chirps
    else:
        raise ValueError("time_unit must be 'second', '15sec', or 'minute'")
    
    temp_pred = model.predict([[chirps_per_min]])[0]
    return temp_pred

 
temp_15sec = predict_temp_from_chirps(20, '15sec')
print(f"20 chirps in 15 seconds -> {temp_15sec:.2f} deg C")


 

joblib.dump(model, 'dolbear_cricket_model.pkl')
print("\nModel saved as 'dolbear_cricket_model.pkl'")
 

plt.figure(figsize=(10,6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training data', color='green')
plt.scatter(X_test, y_test, alpha=0.6, label='Test data', color='blue')
x_range = np.linspace(df['chirps_per_minute'].min(), df['chirps_per_minute'].max(), 100)
y_range = intercept + slope * x_range
plt.plot(x_range, y_range, 'r-', linewidth=2, label='Regression line (Dolbear)')
plt.xlabel('Chirps per minute')
plt.ylabel('Temperature (deg C)')
plt.title('Dolbear\'s law: cricket chirps vs temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
