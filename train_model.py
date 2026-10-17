import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("Loaded features:", X.shape)
print("Loaded labels:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes found:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Neural network: 2 hidden layers (64 and 32 neurons)
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nFinal test accuracy: {test_acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, "style_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\nSaved model, scaler, and label encoder.")