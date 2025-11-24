# ML for Semiconductor Wafer Data Analysis

This project shows an end-to-end workflow for analyzing semiconductor wafer test data.  
It covers data parsing, SQL storage, wafer visualization, synthetic wafer simulation, and unsupervised anomaly detection using KMeans and IsolationForest.

The goal is to demonstrate a realistic engineering workflow that combines data engineering, visualization, and ML — similar to what is used in yield analysis teams in semiconductor companies.

-------------------------------------

## What This Project Does

1. Loads measurement data from CSV or JSON  
2. Stores everything in a structured SQLite database  
3. Generates synthetic wafers with realistic spatial defect patterns  
4. Visualizes wafers using heatmaps and scatter plots  
5. Runs ML models to detect potential defects  
6. Evaluates detection performance with precision, recall, and F1 score  

Everything is reproducible using Python scripts and a Jupyter notebook.

-------------------------------------

## Folder Structure

wafer-ml-project/
src/
parsing/ → CSV/JSON parsers
db/ → SQLite helpers
ml/ → ML pipeline
simulate.py → Synthetic wafer generator
wafer_data.db → SQLite database
figures/ → Saved plots
notebooks/
analysis.ipynb → Main notebook
README.md


-------------------------------------

## How To Run

### 1. Create environment

python -m venv .venv
..venv\Scripts\activate (Windows)
pip install -r requirements.txt


### 2. Generate synthetic wafers

python src/simulate.py


This creates tables:
- devices_sim  
- wafers  

and injects spatial defects (clusters, rings, noisy regions).

### 3. Run ML pipeline


python -m src.ml.pipeline


This:
- extracts features  
- adds a spatial feature (radial distance)  
- runs KMeans  
- runs IsolationForest  
- saves output to devices_sim_ml  
- writes ml_results.csv  
- exports figures to src/figures  

### 4. Open the notebook


notebooks/analysis.ipynb


Inside the notebook you can:
- explore the wafer
- visualize maps
- run grid-search for contamination tuning
- evaluate precision/recall

-------------------------------------

## Key Features

### Wafer Simulation  
- Realistic die grid  
- Gaussian defect clusters  
- Edge-ring patterns  
- Random electrical variation  
- Ground-truth is_defect mask  

### ML Pipeline  
- StandardScaler  
- KMeans clustering  
- IsolationForest anomaly detection  
- Features used:
  - Vth  
  - log10(leakage)  
  - radial distance (spatial feature)

### Database  
Stored in SQLite:
- devices  
- devices_sim  
- devices_sim_ml  
- wafers  

### Visualizations  
- Vth heatmaps  
- Leakage maps  
- Cluster scatter plots  
- Anomaly overlays  
- High-resolution PNG exports  

-------------------------------------

## Evaluation

After running the pipeline you will see metrics such as:

Precision: X.XXX
Recall: X.XXX
F1 Score: X.XXX


Interpretation:
- Precision is usually high (the model avoids false positives)
- Recall depends on how strong or mild the defects are
- Adding radial distance and tuning contamination improves recall

-------------------------------------

## Why This Project Is Useful

- Demonstrates practical ML on real-world style manufacturing data  
- Shows thoughtful feature engineering  
- Uses spatial context (important in wafer analysis)  
- Includes clear visual outputs  
- Easy to walk through during an interview  
- Highlights SQL, Python, ML, and visualization skills at the same time  

-------------------------------------

## Future Improvements

- Add local neighborhood features  
- Try LocalOutlierFactor or OneClassSVM  
- Detect scratches or line-shaped defects  
- Export full wafer reports  

-------------------------------------



