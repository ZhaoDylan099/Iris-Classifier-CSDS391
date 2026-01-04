import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy.stats import mode


# Load data
data = pd.read_csv("irisdata.csv")

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values  


encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- RELU --------------------

# Two hidden layers with 10 neurons each
relu_clf = MLPClassifier(
    hidden_layer_sizes=(10,10),
    activation='relu',
    max_iter=2000,
    random_state=42
)
relu_clf.fit(X_train, y_train)
y_pred = relu_clf.predict(X_test)

print("\nReLU")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ------------------- Other Functions -------------

def train_mlp(X_train, y_train, X_test, y_test, activation):

    # Two hidden layers with 10 neurons each
    clf = MLPClassifier(hidden_layer_sizes=(10,10), activation=activation, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nActivation: {activation}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return y_pred

for act in ['tanh','logistic','identity']:
    train_mlp(X_train, y_train, X_test, y_test, act)

# --------------------- KMEANS --------------------

# Load data
data = pd.read_csv("irisdata.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode class labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply K-Means (3 clusters since 3 classes)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# K-Means cluster assignments
clusters = kmeans.labels_

# Convert cluster labels → true class labels using majority vote
mapped_labels = np.zeros_like(clusters)

for cluster in range(3):
    mask = clusters == cluster
    mapped_labels[mask] = mode(y_encoded[mask], keepdims=True)[0]

# Evaluation
print("\nK-Means Clustering Results")
print("Accuracy:", accuracy_score(y_encoded, mapped_labels))
print(classification_report(y_encoded, mapped_labels, target_names=encoder.classes_))
