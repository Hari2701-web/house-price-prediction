# 🏠 House Price Prediction System

> An end-to-end Machine Learning project that predicts California median house values using **Linear Regression** and **Random Forest** — served via a **FastAPI** backend and a clean **HTML/CSS/JS** frontend.

---

## 📸 Project Overview

| Layer | Technology |
|---|---|
| Data & ML | Python · Pandas · NumPy · scikit-learn |
| Visualisation | Matplotlib · Seaborn |
| API backend | FastAPI · Uvicorn · Pydantic |
| Frontend | HTML5 · CSS3 · Vanilla JavaScript |
| Dataset | [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (scikit-learn) |

The system takes 8 census-block-group features as input and returns a predicted median house value in US dollars.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py       # Load & inspect the dataset
│   └── preprocessing.py     # Feature selection, scaling, splitting
│
├── model/
│   ├── __init__.py
│   ├── train.py             # Train, evaluate, and save models
│   ├── visualize.py         # Generate diagnostic plots
│   ├── best_model.pkl       # Best model (created after training)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── evaluation_results.json
│   └── plots/               # PNG charts (created after training)
│
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
│
├── frontend/
│   └── index.html           # Single-page prediction UI
│
├── notebooks/
│   └── exploration.ipynb    # Step-by-step EDA & model walkthrough
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or newer
- pip

### 1 · Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2 · Create and activate a virtual environment (recommended)

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Step 1 · Train the Model

Run the training script from the **project root**:

```bash
python model/train.py
```

This will:
- Download the California Housing dataset (automatic, no manual download needed)
- Split data into 80% train / 20% test
- Train **Linear Regression** and **Random Forest** models
- Print evaluation metrics (MAE, RMSE, R²) for both
- Save `model/best_model.pkl`, `model/linear_regression.pkl`, `model/random_forest.pkl`
- Save `model/evaluation_results.json`

Expected output (approximate):

```
══ Linear Regression ══
MAE  : 0.5332
RMSE : 0.7458
R²   : 0.5757  (57.6% variance explained)

══ Random Forest ══
MAE  : 0.3271
RMSE : 0.5072
R²   : 0.8050  (80.5% variance explained)

🏆 Best model: Random Forest  (R² = 0.805)
```

### Step 2 · Generate Visualisation Plots (Optional)

```bash
python model/visualize.py
```

Saves 5 PNG plots to `model/plots/`:

| File | Description |
|---|---|
| `01_price_distribution.png` | Raw and log-transformed price histogram |
| `02_correlation_heatmap.png` | Feature correlation matrix |
| `03_feature_importance.png` | Random Forest feature importances |
| `04_actual_vs_predicted.png` | Scatter: true vs predicted prices |
| `05_residual_plot.png` | Residuals vs fitted + distribution |

### Step 3 · Start the FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```

The API is now live at **http://localhost:8000**

- Interactive docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Model info: http://localhost:8000/model-info

### Step 4 · Open the Frontend

Simply open the file in your browser:

```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

Or serve it with Python's built-in server:

```bash
cd frontend
python -m http.server 3000
# then visit http://localhost:3000
```

> ⚠️ The FastAPI server **must** be running on port 8000 before you use the frontend.

---

## 📡 API Reference

### `POST /predict`

Predict the median house value.

**Request body (JSON):**

```json
{
  "MedInc":     3.87,
  "HouseAge":   29.0,
  "AveRooms":   5.43,
  "AveBedrms":  1.09,
  "Population": 1015.0,
  "AveOccup":   2.72,
  "Latitude":   34.05,
  "Longitude":  -118.24
}
```

**Response:**

```json
{
  "predicted_price_100k": 2.3714,
  "predicted_price_usd":  237140.0,
  "model_used":           "RandomForestRegressor",
  "input_features":       { ... }
}
```

**Test with curl:**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "MedInc": 3.87, "HouseAge": 29, "AveRooms": 5.43,
       "AveBedrms": 1.09, "Population": 1015, "AveOccup": 2.72,
       "Latitude": 34.05, "Longitude": -118.24
     }'
```

---

## 📊 Input Features

| Feature | Description | Unit |
|---|---|---|
| `MedInc` | Median household income | ×$10,000 |
| `HouseAge` | Median age of houses in block | Years |
| `AveRooms` | Average rooms per household | Count |
| `AveBedrms` | Average bedrooms per household | Count |
| `Population` | Total block-group population | People |
| `AveOccup` | Average household size | People |
| `Latitude` | Block-group latitude | Degrees |
| `Longitude` | Block-group longitude | Degrees |

**Target:** `MedHouseVal` – median house value in units of **$100,000**
(e.g. model output of `2.5` = $250,000)

---

## 🧪 Running the Notebook

```bash
pip install jupyter
jupyter notebook notebooks/exploration.ipynb
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a Pull Request

---

## 📄 License

MIT © 2025 – free to use for learning and portfolio projects.
