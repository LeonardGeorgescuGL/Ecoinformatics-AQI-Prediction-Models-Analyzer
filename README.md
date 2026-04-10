# 🌍 Bucharest Air Quality Intelligence System (Eco-Informatică Hibridă)

https://ecoinformatics-aqi-prediction-models-analyzer-22yl.streamlit.app/ 
A comprehensive, hybrid eco-informatics dashboard built with **Streamlit**. This application monitors, analyzes, and forecasts the Air Quality Index (AQI) for Bucharest, Romania, by integrating real-time weather and pollution data with advanced Machine Learning time-series models.

## ✨ Key Features

* **Real-Time Data Extraction:** Fetches live environmental and meteorological data via the OpenWeather API.
* **Advanced Forecasting (ML Pipeline):** Compares predictions across multiple models including **Prophet, ARIMA, SARIMA, and XGBoost**.
* **Interactive Visualizations:** Utilizes `Plotly` to render dynamic, interactive charts for pollutants (PM2.5, NO₂, etc.) and AQI trends.
* **Automated Public Policy Recommendations:** Generates actionable, data-driven recommendations for public administration based on traffic, green areas, and industrial parameters.
* **SAS ETL Integration:** Prepared for integration with SAS pipelines for robust data transformation.

## 🛠️ Technology Stack

* **Frontend:** Streamlit, Plotly
* **Backend / ML:** Python, Pandas, Scikit-Learn, Statsmodels, Prophet, XGBoost, SciPy
* **Data Sources:** OpenWeather API

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LeonardGeorgescuGL/Ecoinformatics-AQI-Prediction-Models-Analyzer.git
   cd Ecoinformatics-AQI-Prediction-Models-Analyzer
