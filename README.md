# Airbnb Listings Analysis — Pakistan 🇵🇰

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Inter&size=28&duration=3500&pause=900&color=2F81F7&center=true&vCenter=true&width=900&lines=EDA+%7C+Interactive+Dashboard+%7C+Price+Prediction+ML;City-wise+Insights+and+Interactive+Maps" alt="Typing animation: EDA | Interactive Dashboard | Price Prediction ML" />

<!-- Badges -->
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
<a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-1.x-150458?logo=pandas&logoColor=white" alt="Pandas"></a>
<a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
<a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn"></a>
<a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-visualization-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib"></a>
<a href="https://seaborn.pydata.org/"><img src="https://img.shields.io/badge/Seaborn-stats%20plots-4C8CBF?logo=seaborn&logoColor=white" alt="Seaborn"></a>
<a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-apps-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
<a href="https://plotly.com/python/"><img src="https://img.shields.io/badge/Plotly-interactive-3F4F75?logo=plotly&logoColor=white" alt="Plotly"></a>
<a href="https://python-visualization.github.io/folium/"><img src="https://img.shields.io/badge/Folium-maps-43A047?logo=leaflet&logoColor=white" alt="Folium"></a>
<a href="https://requests.readthedocs.io/"><img src="https://img.shields.io/badge/requests-HTTP-1E90FF?logo=python&logoColor=white" alt="requests"></a>
<a href="https://www.crummy.com/software/BeautifulSoup/"><img src="https://img.shields.io/badge/BeautifulSoup4-HTML%20parse-1D6F42" alt="BeautifulSoup4"></a>
<a href="https://geopy.readthedocs.io/"><img src="https://img.shields.io/badge/geopy-geocoding-2F7B28" alt="geopy"></a>

</div>

<!-- Animated tech strip (as used in other repos) -->
<div align="center" style="margin: 10px 0 2px 0;">
<marquee behavior="scroll" direction="left" scrollamount="6" style="width:100%;">
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" height="38" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="Pandas" height="38" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="NumPy" height="38" />
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="scikit-learn" height="38" />
  <img src="https://matplotlib.org/_static/logo2_compressed.svg" alt="Matplotlib" height="38" />
  <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn" height="38" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/streamlit/streamlit-original.svg" alt="Streamlit" height="38" />
  <img src="https://images.plot.ly/logo/new-branding/plotly-logomark.png" alt="Plotly" height="38" />
  <img src="https://leafletjs.com/docs/images/logo.svg" alt="Leaflet/Folium" height="38" />
</marquee>
</div>

---

## Overview

Analyze Airbnb listings across Pakistan to understand pricing dynamics, city-wise patterns, and build ML models for price prediction. The project includes:
- Exploratory Data Analysis (EDA)
- Interactive Streamlit dashboard
- Geographic visualization with Folium
- Machine learning models for price prediction (Linear Regression, Random Forest)

Major cities covered: Islamabad, Lahore, Karachi, Rawalpindi.

---

## Goals

- Price prediction using ML (Linear Regression, Random Forest)
- EDA of listing features, city-wise trends, and distributions
- Market insights: average prices, demand hotspots, outliers

---

## Datasets

- Airbnb scraped listings: airbnb_listings_pakistan.csv
- Consolidated workbook: Combined.xlsx
- Cleaned feature set: df_combined_cleaned.csv
- City aggregates: city_analysis_major.csv
- Optional external: Pakistan real-estate (via Kaggle) for enrichment

Data preprocessing:
- Converted prices (PKR/$ → numeric)
- Handled missing values (median/most-frequent)
- Extracted city/location features
- Dropped high-null columns and standardized schema

---

## Key Findings

- Significant price variation by city; Islamabad and Lahore tend to be higher.
- Outliers present with extremely high prices (influencing mean).
- Random Forest outperforms Linear Regression on test splits but variance remains unexplained.
- Improvement ideas:
  - Feature engineering: reviews, amenities, proximity to landmarks, superhost effects
  - Hyperparameter tuning
  - Advanced models: XGBoost, LightGBM

---

## Visual Gallery

- Price distribution and boxplots by room type
- City-wise average price bars
- Outlier detection via IQR
- Actual vs Predicted price scatter with reference line
- Interactive Pakistan map with price-based color coding

Tip: Place your exported images into docs/ and they will render below.

<div align="center">
  <img src="docs/price_distribution.png" alt="Price distribution" width="45%"/>
  <img src="docs/boxplot_prices.png" alt="Boxplot by room type" width="45%"/><br/>
  <img src="docs/citywise_avg_price_bar.png" alt="City-wise average price" width="45%"/>
  <img src="docs/prediction_vs_actual.png" alt="Actual vs Predicted" width="45%"/>
</div>

Interactive map (HTML): airbnb_pakistan_map_colored.html

---

## Interactive Dashboard (Streamlit)

A full-featured app with:
- Overview: KPIs, dataset preview, room-type distribution
- Price Analysis: histograms, boxplots, city comparisons, price vs rating
- Geographic Analysis: interactive Folium map and city counts
- Model Predictions: on-the-fly training (sklearn pipeline), metrics, and interactive predictor
- Key Insights: business takeaways and recommendations

Open the app:
```bash
pip install -r requirements.txt
streamlit run airbnb_dashboard.py
```

Relevant file: airbnb_dashboard.py

---

## Quickstart

1) Clone
```bash
git clone https://github.com/tahahasan01/Airbnb_Listings_Analysis.git
cd Airbnb_Listings_Analysis
```

2) Install
```bash
pip install -r requirements.txt
```

3) Notebooks
- Open the analysis notebook in Jupyter: Airbnb_Analysis (1).ipynb

4) Dashboard
```bash
streamlit run airbnb_dashboard.py
```

5) Map (Folium)
```bash
python visualize_airbnb_map.py
# Output: airbnb_pakistan_map_colored.html
```

---

## Machine Learning

- Features: roomType, stars, city, isHostedBySuperhost, location/lat, location/lng, numberOfGuests
- Pipeline:
  - ColumnTransformer with SimpleImputer (median for numeric) + OneHotEncoder (categorical)
  - Models: Linear Regression, Random Forest Regressor (notebook)
- Metrics: MAE, MSE, R²
- Visualization: Actual vs Predicted with identity line

---

## Tech Stack

- Data: pandas, numpy, openpyxl
- Viz: matplotlib, seaborn, plotly
- Maps: folium, geopy
- ML: scikit-learn (pipelines, preprocessing, train-test split, Linear Regression, Random Forest)
- App: streamlit
- Scraping: requests, beautifulsoup4, lxml
- Optional: Kaggle dataset ingestion for enrichment

---

## Repository Structure

```
Airbnb_Listings_Analysis/
├── Airbnb_Analysis (1).ipynb
├── airbnb_dashboard.py
├── visualize_airbnb_map.py
├── airbnb_listings_pakistan.csv
├── df_combined_cleaned.csv
├── city_analysis_major.csv
├── Combined.xlsx
├── airbnb_pakistan_map_colored.html
├── requirements.txt
└── docs/                      # (optional) export plots here for README
```

---

## Usage Notes

- The dashboard caches data for responsive navigation.
- The Folium map colors markers by price buckets.
- If using external Kaggle data, ensure proper merging and re-run cleaning scripts; optionally add kagglehub to your environment.

---

## Acknowledgments

- Airbnb public listings (for research/educational purposes)
- Python open-source ecosystem
- Folium/Leaflet for interactive mapping
- scikit-learn for ML pipelines

---

<div align="center">
Made with ❤️ in Pakistan • by <a href="https://github.com/tahahasan01">tahahasan01</a>
</div>
