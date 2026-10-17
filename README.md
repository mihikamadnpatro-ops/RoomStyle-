# 🎨 RoomStyle AI

**Interior style classifier and paint colour recommender.** Upload a photo of a room, and the app predicts its interior design style and suggests matching paint colours.

---

## Overview

Paint and home-decor brands increasingly use digital tools to help customers choose colours that suit their space. RoomStyle AI is a simplified version of that idea. It takes a room photo, predicts the design style using a neural network, and recommends colours that tend to suit that style.

The project covers the main stages of a machine learning pipeline: data collection, feature extraction, model training, evaluation, and an interactive web app built with Streamlit.

## How it works

1. **Data:** Room photos collected from Unsplash, sorted into one folder per style.
2. **Feature extraction:** Each image is resized to 64×64 and converted into a normalised colour histogram (16 bins × 3 RGB channels = 48 features).
3. **Preprocessing:** Features are standardised with `StandardScaler`, and labels are encoded with `LabelEncoder`.
4. **Model:** A feedforward neural network (`MLPClassifier`, hidden layers of 64 and 32 neurons, ReLU activation) outputs a probability for each style.
5. **Evaluation:** Stratified 80/20 train-test split, with accuracy and a per-class classification report.
6. **App:** A Streamlit interface that shows the uploaded photo, the predicted style, the confidence, suggested paint colours, and the probability for every style.

## Styles covered

| Style | Label name in code |
|---|---|
| Bohemian | `bohemain` |
| Industrial | `industrial` |
| Minimalist | `minimalist` |
| Traditional | `traditional` |

> Note: the Bohemian label is spelled `bohemain` in the code because the dataset folder was named that way.

## Tech stack

Python, NumPy, Pillow, scikit-learn, Joblib, Streamlit.

## Project structure

```
roomstyle-ai/
├── app.py                  # Streamlit app
├── extract_features.py     # Turns images into colour-histogram features
├── train_model.py          # Trains and evaluates the model, saves artefacts
├── images.py               # Quick check of image counts per folder
├── style_classifier.pkl    # Trained neural network
├── scaler.pkl              # Fitted StandardScaler
├── label_encoder.pkl       # Fitted LabelEncoder
├── requirements.txt
└── README.md
```

The raw image dataset is not included in this repository.

## Run it locally

```bash
# 1. Clone the repo
git clone https://github.com/[your-username]/roomstyle-ai.git
cd roomstyle-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python -m streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`. Upload a room photo to see the prediction.

### Retrain the model (optional)

You need your own images in `raw_images/<style_name>/`, with one folder per style.

```bash
python extract_features.py
python train_model.py
```

## Results

| Metric | Value |
|---|---|
| Classes | [add number] |
| Total images | [add number] |
| Test accuracy | [add your final accuracy] |

Replace the bracketed values with the numbers printed by `train_model.py`. With a small dataset and colour-only features, accuracy is modest, and the per-class report shows which styles the model confuses most.

## Limitations

- **Colour histograms are a weak feature for style.** They capture the overall colour mix but nothing about shape, texture, layout or objects, so two rooms with similar palettes can belong to different styles.
- **Small dataset.** Only a few hundred images per class, so results vary between train-test splits.
- **Paint suggestions are illustrative.** The colour names are hand-mapped suggestions per style. They are not official shades from any paint brand.
- **Class removed.** A "coastal" class was dropped because it had too few images and the model could not learn it.

## Future improvements

- Replace colour histograms with features from a pretrained CNN (transfer learning).
- Add dimensionality reduction (PCA or LDA) and compare it against raw features.
- Collect more images and use data augmentation.
- Add cross-validation and a baseline comparison (for example, logistic regression).
- Map styles to real paint shades from a brand's public colour catalogue.
- Deploy the app publicly (Streamlit Community Cloud or Hugging Face Spaces).

## Dataset and licensing

Images were collected from [Unsplash](https://unsplash.com), which allows free use under the Unsplash licence. The images are used only for training and are not redistributed here.

## Author

**Mihika Patro**
B.Tech Data Science, NMIMS MPSTME, Mumbai
[LinkedIn](https://linkedin.com/in/mihikapatro-54525331a) · [GitHub](https://github.com/mihikamadnpatro-ops)
