# GeoLife Transportation Mode Classification (CET 522 Project)

This project uses the Microsoft Research **GeoLife Trajectories 1.3** dataset
with transportation–mode labels for a subset of users. The pipeline covers:

- **Data management**: convert raw trajectories and label intervals into
  trip segments (`segments`) and engineered features, stored in a local
  SQLite database.
- **ML analysis**: train a baseline transportation–mode classifier using
  handcrafted features.
- **Visualization & app**: provide an interactive Streamlit web app for
  exploring the data and model predictions.

## Repository layout

- `pipeline/`
  - `build_db.py`: build the SQLite DB from raw GeoLife data
    (`segments`, `segment_points`, `segment_features`).
  - `train_model.py`: train the classifier (MLP) and save the model and metrics.
- `app/`
  - `app.py`: Streamlit application entry point.
- `src/`: shared utilities for parsing GeoLife data, feature engineering,
  and SQLite helpers.
- `data/`: generated SQLite database (default location).
- `models/`: trained model and metrics files.

## Quick start

1) Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Download **GeoLife Trajectories 1.3** from the [official site](https://www.microsoft.com/en-us/download/details.aspx?id=52367), then set `DEFAULT_GEOLIFE_ROOT` in `src/constants.py` to your data folder.

3) Build the database (uses only users that have a `labels.txt` file):

```bash
python pipeline/build_db.py
```

4) Train the model:

```bash
python pipeline/train_model.py
```

5) Run the Streamlit app:

```bash
streamlit run app/app.py
```

## Data license note

The GeoLife dataset is released **for non‑commercial use only**.
Do **not** publicly redistribute the original data or any derived data.

