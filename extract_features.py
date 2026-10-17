import os
import numpy as np
from PIL import Image

def extract_features(img_path, size=(64, 64)):
    img = Image.open(img_path).convert("RGB").resize(size)
    img_array = np.array(img)

    hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256))
    hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256))
    hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256))

    features = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    features = features / features.sum()
    return features

data_dir = "C:/Users/mihika/OneDrive/Desktop/data/raw_images"  # update to your actual path
X = []
y = []

for style in os.listdir(data_dir):
    style_path = os.path.join(data_dir, style)
    for fname in os.listdir(style_path):
        fpath = os.path.join(style_path, fname)
        try:
            feat = extract_features(fpath)
            X.append(feat)
            y.append(style)
        except Exception as e:
            print(f"Skipped {fpath}: {e}")

X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
np.save("X_features.npy", X)
np.save("y_labels.npy", y)