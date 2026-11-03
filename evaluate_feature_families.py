import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

d = np.load("localisation_features.npz")

Y = d["Y"]
groups = d["groups"]

features = {
    "Amplitude": ["amplitude"],
    "Complex": ["complex"],
    "Phase": ["phase"],
    "RSSI": ["rssi"],
    "Temporal": ["temporal"],
    "Doppler": ["doppler"],
    "Correlation": ["correlation"],
    "Amplitude + Phase": ["amplitude", "phase"],
    "Amplitude + Complex": ["amplitude", "complex"],
    "Amplitude + Phase + RSSI": ["amplitude", "phase", "rssi"],
    "Amplitude + Phase + Temporal": [
        "amplitude", "phase", "temporal"
    ],
    "All features": [
        "amplitude", "complex", "phase", "rssi",
        "temporal", "doppler", "correlation"
    ]
}

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "KNN 3": KNeighborsRegressor(
        n_neighbors=3, weights="distance"
    ),
    "KNN 5": KNeighborsRegressor(
        n_neighbors=5, weights="distance"
    ),
    "Ridge": Ridge(alpha=10.0),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
}

cv = LeaveOneGroupOut()
results = []

for feature_name, keys in features.items():

    X = np.concatenate([d[k] for k in keys], axis=1)

    print("\nFEATURE SET:", feature_name, X.shape)

    for model_name, model in models.items():

        predictions = np.zeros_like(Y)

        for train, test in cv.split(X, Y, groups):

            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                model
            )

            estimator.fit(X[train], Y[train])
            predictions[test] = estimator.predict(X[test])

        # One position prediction per held-out recording
        recording_errors = []

        for group in np.unique(groups):
            idx = np.where(groups == group)[0]

            actual = np.mean(Y[idx], axis=0)
            predicted = np.mean(predictions[idx], axis=0)

            recording_errors.append(
                np.linalg.norm(actual - predicted) * 100
            )

        mean_error = np.mean(recording_errors)
        median_error = np.median(recording_errors)

        results.append((
            mean_error, median_error, feature_name, model_name
        ))

        print(
            model_name,
            "| Mean:", round(mean_error, 2), "cm",
            "| Median:", round(median_error, 2), "cm"
        )

print("\n\nOVERALL RANKING")
for mean_error, median_error, feature_name, model_name in sorted(results):
    print(
        round(mean_error, 2), "cm |",
        round(median_error, 2), "cm |",
        feature_name, "|", model_name
    )
