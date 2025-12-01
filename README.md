# 🏠 Munich Airbnb Price & Occupancy Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Data Mining and Machine Learning Project (2024-2025)**  
> **University of Pisa**  
> **Authors:** Reza Almassi & Jad Mahmoud  
> **Professors:** J.L. Corcuera Bárcena & F. Marcelloni

---

## 📌 Project Overview
This project analyzes the Munich short-term rental market using the **Inside Airbnb** dataset. We applied a full Data Mining pipeline—from Exploratory Data Analysis (EDA) to Advanced Modeling—to solve two key problems:

1.  **Price Prediction (Regression):** Predicting the nightly price of a listing based on its features.
2.  **Occupancy Prediction (Classification):** Predicting whether a listing will be booked or available on a specific date.

The goal was to demonstrate rigorous data cleaning, feature engineering, and model optimization techniques.

---

## 📂 Dataset
The data is sourced from [Munich Airbnb Data on Kaggle](https://www.kaggle.com/datasets/chriskue/munich-airbnb-data).
*   **listings.csv:** Detailed attributes for ~11,000 listings (rooms, amenities, host info).
*   **calendar.csv:** Daily availability and price data.

---

## ⚙️ Methodology & Pipeline

### 1. Data Cleaning & Preprocessing
*   **Currency Conversion:** Parsed string prices (e.g., "$1,200.00") into numeric floats.
*   **Outlier Removal (Statistical):** Applied the **IQR (Interquartile Range)** method to filter extreme price outliers. We focused on the valid market range (approx. €20 - €300).
*   **Imputation:** Handled missing values using:
    *   *Median* for structural features (bathrooms, beds).
    *   *Flagging* for missing review scores (capturing "new listing" signal).

### 2. Advanced Feature Engineering
To improve model performance beyond the baseline, we engineered several custom features:
*   **📍 Distance to Center:** Used `geopy` to calculate the geodesic distance (km) from each listing to **Marienplatz** (Munich City Center).
*   **⏳ Host Tenure:** Calculated the number of years a host has been active.
*   **📶 Amenities Parsing:** Extracted binary features for high-value amenities (WiFi, Kitchen) and total amenity count.
*   **🗓️ Temporal Features:** Extracted Month and Day of Week for the occupancy task.

### 3. Model Selection & Tuning
We compared multiple algorithms using **5-Fold Cross-Validation** to ensure stability:
*   **Linear Models:** Lasso Regression.
*   **Tree-Based Models:** Random Forest, XGBoost.

We used **GridSearchCV** to fine-tune hyperparameters (Learning Rate, Max Depth, N Estimators).

---

## 📊 Results

### Task 1: Price Prediction (Regression)
We significantly reduced the error by moving from a baseline Linear Regression to a tuned XGBoost model.

| Model | MAE (Mean Absolute Error) | Notes |
| :--- | :--- | :--- |
| **Baseline (Linear)** | €54.86 | High error due to outliers & non-linearity. |
| **Random Forest (Initial)** | €52.23 | Better, but struggled with luxury units. |
| **XGBoost (Tuned + IQR Filter)** | **€32.58** | **Best Performance.** |

*Key Insight:* The most important predictors for price were **Accommodates** (Capacity), **Distance to Center**, and **Room Type**.

### Task 2: Occupancy Prediction (Classification)
The dataset was highly imbalanced (most dates are "Booked"). A standard model had high accuracy but 3% Recall for the "Available" class.

*   **Solution:** We applied **Class Weighting (`class_weight='balanced'`)**.
*   **Outcome:** 
    *   **Recall (Available Class):** Improved from **0.03** → **0.64**.
    *   The model can now successfully identify available dates, making it a viable tool for users.

---

## 📈 Visualizations
*(Note: Visuals generated in the Jupyter Notebook)*

1.  **Feature Importance:** Confirmed that `distance_center_km` and `accommodates` are the top drivers.
2.  **Residual Analysis:** Shows error distribution is centered at 0, confirming unbiased predictions for standard apartments.

---

## 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost geopy
    ```
3.  Place `listings.csv` and `calendar.csv` in the root folder.
4.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook Munich_Airbnb_Analysis.ipynb
    ```

---

## 📜 Dependencies
*   Python 3.x
*   Pandas / NumPy
*   Scikit-Learn
*   XGBoost
*   Geopy
*   Matplotlib / Seaborn
