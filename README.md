# Student Productivity Dashboard

An interactive Dash app for exploring how students allocate daily time (study, sleep, and phone use) and how that relates to productivity, focus, and stress.  
The dashboard is designed for visual exploration with coordinated filters and multiple chart views.

## Live Demo

- Online demo: https://<your-demo-url>

## What You Can Explore

- **Global filters** for Gender, Age Range, Main Activity Time Range, and Productivity Score Range.
- **Ternary Productivity Heatmap**: average productivity under different time-allocation combinations.
- **Ternary Density Heatmap**: where time-allocation patterns are most concentrated.
- **Three density panels**:
  - Study Time vs Productivity
  - Sleep Time vs Productivity
  - Phone Time vs Productivity
- **Focus Score Distribution**: histogram with KDE curve.
- **Stress Level Distribution**: density-based histogram.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── student_productivity_distraction_dataset_20000.csv
└── assets/
```

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Then open: `http://127.0.0.1:8050`

## Data

The app expects a CSV file named `student_productivity_distraction_dataset_20000.csv` in the project root, including these columns:

- `age`
- `gender`
- `study_hours_per_day`
- `sleep_hours`
- `phone_usage_hours`
- `productivity_score`
- `focus_score`
- `stress_level`

Rows with non-positive total main activity time (`study + sleep + phone`) are excluded.

## Tech Stack

- Python
- Dash
- Plotly
- Pandas
- NumPy
- SciPy

## Deployment

This project can be deployed on platforms like Render.  
For production, serve the Dash app with Gunicorn (example): `gunicorn app:server`.

## License

For educational and demo purposes.
