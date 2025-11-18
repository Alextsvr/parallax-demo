# Parallax CTS — Shooting Analytics Demo

This repository contains a demonstration prototype of AI-powered shooting analytics.
The goal is to showcase how training video and shooting-series data can be transformed into actionable insights for shooters and instructors.

The demo includes:
- Synthetic shooting data generation
- Exploratory analytics and visualizations
- A baseline ML model predicting next-series accuracy
- Computer Vision pose analysis
- A Streamlit dashboard with PDF export

---

## Project Overview

Parallax CTS — Shooting Analytics Demo demonstrates a full pipeline:

1. Data ingestion  
   Synthetic dataset of shooters, sessions, series, and shot coordinates.

2. Analytics dashboard  
   Heatmaps, accuracy trends, shooter/session statistics.

3. Machine Learning module  
   Predicts accuracy of the next shooting series using LightGBM.

4. Computer Vision module  
   Detects pose landmarks and evaluates shooter stability using MediaPipe.

5. PDF reporting  
   Generates a compact instructor-friendly report.

This demo is designed as a proof of concept (POC) and blueprint for future expansion into a production-grade analytics platform.

---

## Repository Structure

```
parallax-demo/
├─ data/
│  └─ shooting_synthetic.csv
├─ notebooks/
│  └─ 01_EDA_and_model.ipynb
├─ src/
│  ├─ data_gen.py
│  ├─ features.py
│  ├─ modeling.py
│  └─ cv/
│     └─ pose_analysis.py
├─ app/
│  └─ streamlit_app.py
├─ reports/
├─ slides/
│  └─ demo_parallax.pptx
├─ requirements.txt
└─ README.md
```

---

## Tech Stack

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn, lightgbm
- plotly, matplotlib, seaborn, shap
- opencv-python, mediapipe
- streamlit
- reportlab
- optional: docker, fastapi

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Alextsvr/parallax-demo.git
cd parallax-demo
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:

Windows:
```powershell
venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Generating Synthetic Data

Run the data generation script:

```bash
python src/data_gen.py
```

This creates:

```
data/shooting_synthetic.csv
```

The dataset includes:
- shooter_id, session_id, series_id  
- shot coordinates (x, y)  
- timestamps  
- weapon type  
- distance  
- hit/miss  
- time since start  

---

## EDA and ML Notebook

Open the main notebook:

```
notebooks/01_EDA_and_model.ipynb
```

It includes:
- Per-shooter accuracy  
- Heatmaps of shot density  
- Session trends  
- Feature engineering  
- LightGBM baseline model  
- SHAP feature importance  

---

## ML Module

The ML module (src/modeling.py) performs:
- Aggregation to per-series features  
- Train/test split by shooter  
- LightGBM regression  
- Metrics (MAE, RMSE, R2)  
- SHAP summary plot  
- Test predictions export  

---

## Computer Vision Pose Analysis

The CV script (src/cv/pose_analysis.py):
- Reads an input training video  
- Uses MediaPipe Pose to detect landmarks  
- Computes a stability score  
- Draws annotations and metrics  
- Saves an annotated video  

Example:
```bash
python src/cv/pose_analysis.py --input input.mp4 --output annotated.mp4
```

---

## Streamlit Dashboard

Launch the demo dashboard:

```bash
streamlit run app/streamlit_app.py
```

Features:
- Shooter and session selector  
- Shot heatmap  
- Accuracy trends  
- ML prediction with SHAP bar chart  
- Embedded annotated video  
- PDF export  

---

## Presentation Materials

The folder slides/demo_parallax.pptx contains:
- Overview  
- Data to model to CV pipeline  
- Screenshots  
- Business value  
- Pilot proposal  

---

## Roadmap

- Improve synthetic data realism  
- Add leaderboard-style analytics  
- Implement FastAPI backend  
- Extend pose analysis  
- Multi-camera support  

---

## License

MIT License.

---

## Contact

Aleksandrs Cveckovskis  
Data Scientist / ML Engineer  
Email: aleksandrs.cveckovskis@gmail.com
