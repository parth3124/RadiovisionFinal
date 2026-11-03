import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

data = np.load("localisation_dataset.npz")

X = data["X"]
Y = data["Y"]
names = data["names"]

# Standardise each CSI feature across the 25 positions
X_scaled = StandardScaler().fit_transform(X)

# Pairwise CSI distances
D = euclidean_distances(X_scaled)

print("NEAREST CSI FINGERPRINTS")
print()

correct_neighbour = 0
neighbour_errors = []

for i in range(len(X)):
    distances = D[i].copy()
    distances[i] = np.inf

    nearest = np.argmin(distances)
    spatial_error = np.linalg.norm(Y[i] - Y[nearest])

    neighbour_errors.append(spatial_error)

    if spatial_error < 0.20:
        correct_neighbour += 1

    print(
        names[i],
        "->", names[nearest],
        "| Physical distance:",
        round(spatial_error * 100, 1), "cm"
    )

print("\nSUMMARY")
print("Nearest-neighbour physical distance:")
print("Mean:", round(np.mean(neighbour_errors) * 100, 2), "cm")
print("Median:", round(np.median(neighbour_errors) * 100, 2), "cm")
print("Within 20 cm:", correct_neighbour, "/", len(X))
