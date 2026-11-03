import numpy as np

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

data = np.load("localisation_dataset.npz")

X = data["X"]
Y = data["Y"]
names = data["names"]

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),

    "KNN (3)": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=3, weights="distance")
    ),

    "KNN (5)": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5, weights="distance")
    ),

    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42
    )
}

loo = LeaveOneOut()

print("LOCALISATION MODEL EVALUATION")
print("Samples:", len(X))
print()

results = []

for name, model in models.items():

    predictions = cross_val_predict(model, X, Y, cv=loo)

    errors = np.linalg.norm(predictions - Y, axis=1)

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    results.append((name, mean_error))

    print(name)
    print("  Mean error:  ", round(mean_error * 100, 2), "cm")
    print("  Median error:", round(median_error * 100, 2), "cm")
    print("  Maximum:     ", round(max_error * 100, 2), "cm")
    print()

print("PREDICTIONS FROM BEST MODEL")

best_name = min(results, key=lambda x: x[1])[0]
best_model = models[best_name]

predictions = cross_val_predict(best_model, X, Y, cv=loo)

print("Best model:", best_name)
print()

for i in range(len(Y)):
    error = np.linalg.norm(predictions[i] - Y[i]) * 100

    print(
        names[i],
        "| Actual:", np.round(Y[i], 3),
        "| Predicted:", np.round(predictions[i], 3),
        "| Error:", round(error, 2), "cm"
    )
