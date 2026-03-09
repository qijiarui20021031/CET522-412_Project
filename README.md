# GeoLife Transportation Mode Classification (CET 522 Project)

This project uses the Microsoft Research **GeoLife Trajectories 1.3** dataset
to classify transportation modes (walk, bike, bus, car) from GPS trajectories.
The pipeline covers data management, ML modeling, and an interactive Streamlit app.

## Features

- **Data management**: Convert raw trajectories and label intervals into trip segments, compute features (speed, acceleration, dwell time, etc.), and store everything in SQLite.
- **ML analysis**: Train a baseline MLP classifier with handcrafted segment-level features.
- **Streamlit app**: Interactive visualization, model results, segment-level prediction demo, and accuracy explorer by user/mode.

## Prerequisites

- **Python 3.10+** (tested on 3.12)
- **GeoLife Trajectories 1.3** dataset (download separately, see below)

---

## 1. Clone and environment setup

```bash
git clone https://github.com/qijiarui20021031/CET522-412_Project.git
cd CET522-412_Project
```

Create a virtual environment and install dependencies. **Use the project’s venv instead of conda base** to avoid NumPy/sklearn conflicts:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Data setup

### Download GeoLife Trajectories 1.3

1. Go to the [official GeoLife download page](https://www.microsoft.com/en-us/download/details.aspx?id=52367).
2. Download and extract the archive.
3. Place the extracted folder inside the project’s `data/` directory so that the path is:

   ```
   CET522-412_Project/
   └── data/
       └── Geolife Trajectories 1.3/
           └── Data/
               ├── 000/
               │   ├── labels.txt
               │   └── Trajectory/
               │       └── *.plt
               ├── 001/
               └── ...
   ```

Only users with a `labels.txt` file are used; others are skipped.

### Alternative data location

If you put GeoLife elsewhere, edit `src/constants.py` and set:

```python
DEFAULT_GEOLIFE_ROOT = Path("/your/path/to/Geolife Trajectories 1.3")
```

---

## 3. Build the database

This step reads raw trajectories and labels, cuts them into segments, computes features, and writes to SQLite:

```bash
python pipeline/build_db.py
```

Output: `data/geolife.sqlite` (the database file).

---

## 4. Train the model

Train the baseline MLP and save model and metrics:

```bash
python pipeline/train_model.py
```

By default this uses Wandb for logging. To skip Wandb:

```bash
python pipeline/train_model.py --no-wandb
```

Output: `models/mode_classifier.joblib` and `models/metrics.json`.

---

## 5. Run the Streamlit app

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`).

### App pages

| Page | Description |
|------|-------------|
| Background & motivation | Project overview and research questions |
| Data & database | Table counts, schema, mode distribution |
| Model results | Confusion matrix and classification report |
| Segment playground | Select a segment, preview trajectory on map, predict mode |
| Accuracy explorer | Filter by user/mode and view accuracy breakdown |
| Summary | Project summary |

---

## Repository layout

| Path | Description |
|------|-------------|
| `pipeline/build_db.py` | Build SQLite DB from raw GeoLife data |
| `pipeline/train_model.py` | Train MLP and save model/metrics |
| `app/app.py` | Streamlit app entry point |
| `src/` | Geoparsing, feature engineering, DB helpers |
| `data/` | Put GeoLife here; `geolife.sqlite` is generated here |
| `models/` | Trained model and metrics (after training) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"numpy.dtype size changed"** or **"NumPy 1.x cannot run in NumPy 2.x"** | Use the project’s `.venv` (not conda base). Run `pip install -r requirements.txt --force-reinstall`. |
| **"Model not found"** / empty `models/` | Run `python pipeline/train_model.py` before starting the app. |
| **"FileNotFoundError"** for GeoLife | Ensure `data/Geolife Trajectories 1.3/` exists or update `DEFAULT_GEOLIFE_ROOT` in `src/constants.py`. |
| **InconsistentVersionWarning** (sklearn) | Model was trained with a different sklearn version. Re-train with `python pipeline/train_model.py`. |

---

## Data license note

The GeoLife dataset is released **for non‑commercial use only**.  
Do **not** publicly redistribute the original data or derived data.
