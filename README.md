# Student Productivity Dashboard

An interactive dashboard for exploring how student time allocation, focus, and stress relate to productivity.

## What This Dashboard Helps You Explore

This project is built for visual analysis, not model training or engineering deep-dives.  
You can quickly compare student groups and spot patterns through a shared set of global filters.

### Global Filters

- Gender
- Age range
- Main activity time range
- Productivity score range

All charts update together after filtering.

### Visualizations

1. **Main Activity Time Allocation vs Productivity**  
   Shows which combinations of study, sleep, and phone time are linked to higher or lower productivity.
2. **Distribution of Main Activity Time Allocation**  
   Shows where most students are concentrated in the study/sleep/phone balance space.
3. **Study Time vs Productivity**  
   Helps you see how productivity changes as study hours increase.
4. **Sleep Time vs Productivity**  
   Helps you compare low-sleep vs balanced-sleep groups.
5. **Phone Time vs Productivity**  
   Helps you inspect how heavier phone use aligns with productivity.
6. **Focus Score Distribution**  
   Shows how focus levels are distributed for the current filtered group.
7. **Stress Level Distribution**  
   Shows whether the filtered group is skewed toward low, medium, or high stress.

## Quick Online Access

To quickly experience the dashboard, simply visit:
https://student-productivity-dashboard.onrender.com

## Local Installation

### Prerequisites

- Python 3.10+

### Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

On Windows (PowerShell), activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Open:

`http://localhost:8050`

### Quick Troubleshooting

- If install fails, upgrade pip first: `python -m pip install --upgrade pip`
- If port `8050` is busy, run with another port:
  - macOS/Linux: `PORT=8060 python app.py`
  - Windows (PowerShell): `$env:PORT=8060; python app.py`

## Deploy to Render (Public URL)

This repo already includes a Render Blueprint file: `render.yaml`.

1. Push this project to GitHub.
2. In Render, click **New +** and choose **Blueprint**.
3. Connect your GitHub repo.
4. Render will read `render.yaml` automatically.
5. Click deploy and wait for build/start to complete.
6. Open the generated public URL.

Notes:

- The service is configured as a Python Web Service.
- On the free plan, the first request after inactivity may be slow (cold start).

## How to Use the Public Render App

1. Open your Render public link in a browser.
2. Set global filters first (gender, age, activity time, productivity range).
3. Read the top two allocation views to understand overall behavior.
4. Use the three time-vs-productivity panels for direct comparisons.
5. Use focus and stress distributions to validate whether patterns are consistent.
6. Share the same URL for demos, reviews, or class presentations.

## Data

The dashboard reads the bundled dataset:

`student_productivity_distraction_dataset_20000.csv`
