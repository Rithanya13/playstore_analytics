---
title: "Predictive Analytics on Google Play Store Apps"
author: "Rithanya Chandran"
date: "May 5, 2025"
---

# 1. Introduction & Dataset Description

## 1.1 Background  
The Google Play Store hosts millions of Android applications, and developers continually compete to maximize downloads, user satisfaction, and revenue. Analyzing factors like review volume, recency of updates, and price can inform strategic decisions around product roadmaps, marketing spend, and pricing models. This project applies an end‑to‑end predictive analytics workflow to a real‑world dataset of Android apps, demonstrating data cleaning, exploratory analysis, modeling, and ethical reflection.

## 1.2 Dataset Selection & Rationale  
- **Dataset:** Google Play Store Apps  
- **Source:** Kaggle “lava18/google-play-store-apps”  
- **Link:** https://www.kaggle.com/datasets/lava18/google-play-store-apps  

This dataset of ~10,000 apps spans 34 categories and includes mixed data types (numerical installs and reviews, text genres, prices with “$” prefixes, “Varies with device” sizes, and dates). Its real‑world noise and variety make it ideal for practicing parsing, feature engineering, and modeling tasks relevant to stakeholders seeking insights into download drivers, rating predictions, and pricing strategies.

# 2. Exploratory Data Analysis (EDA)

## 2.1 Data Overview  
- **Total apps:** 9,660  
- **Missing Ratings:** 1,474 (~15.3%)  
- **Free vs. Paid:** 88% free, 12% paid  
- **Rating:** Mean = 4.18, Median = 4.30  
- **Installs:** Median ≈ 50,000; range up to millions  

## 2.2 Univariate Analysis  
**Figure 1.** Distribution of log(Installs)  
![Distribution of log(Installs)](visualizations/histogram_log_installs.png)  

**Figure 2.** Top 10 App Categories  
![Top 10 App Categories](visualizations/bar_top_categories.png)  

## 2.3 Bivariate Analysis  
**Figure 3.** Reviews vs. Installs (log–log)  
![Reviews vs Installs](visualizations/scatter_reviews_vs_installs.png)  

**Figure 4.** Rating by Top 5 Categories  
![Rating by Category](visualizations/boxplot_rating_by_category.png)  

**Figure 5.** Feature Correlation Matrix  
![Feature Correlation Matrix](visualizations/heatmap_correlations.png)  

## 2.4 Key Insights  
1. `log_Reviews` and `log_Installs` are strongly correlated (ρ≈0.96).  
2. Free apps dominate volume; paid apps are only 12%.  
3. More recent updates modestly increase installs.

# 3. Data Cleaning & Preprocessing

## 3.1 Parsing & Conversion  
- **Reviews:** Handled “M”/“k” suffixes, commas → integer  
- **Installs:** Stripped “+” and commas → integer  
- **Price:** Removed “$”, converted to float, filled NAs with 0  
- **Size:** Converted “M” and “k” units to MB; “Varies with device” → missing  
- **Last Updated:** Parsed to datetime  

## 3.2 Missing Values & Duplicates  
- Dropped duplicates on (App, Category, Last Updated)  
- Imputed missing `Rating` with median (4.30)  

## 3.3 Feature Engineering  
- **log_Installs**, **log_Reviews** for skew reduction  
- **is_free** binary flag  
- **days_since_update** recency in days  
- **One‑hot encoding** of Category & Primary Genre  
- **price_vs_global_median** elasticity signal

# 4. Business Questions

1. **Download‑Volume Drivers**  
   _Which app attributes most influence install counts?_

2. **Rating Prediction**  
   _Can we predict an app’s average user rating from its features?_

3. **Price Optimization**  
   _What price maximizes total predicted revenue, accounting for price sensitivity?_

# 5. Predictive Modeling

## 5.1 Q1 – Installs Prediction  
- **Models:** Linear Regression, Random Forest  
- **Split:** 80% train / 20% test  
- **Metrics:** R², RMSE  

| Model             | R² (Test) | RMSE (Test) |
|-------------------|-----------|-------------|
| Linear Regression | 0.9332    | 1.1543      |
| Random Forest     | 0.9341    | 1.1468      |

**Figure 6.** RF: Actual vs Predicted Installs  
![RF: Actual vs Predicted Installs](visualizations/scatter_installs_actual_vs_predicted.png)  

**Figure 7.** Top 10 Feature Importances  
![Feature Importances](visualizations/bar_feature_importances_installs.png)  

## 5.2 Q2 – Rating Prediction  
- **Models:** Linear Regression, Random Forest  
- **Metrics:** R², RMSE  

| Model             | R² (Test) | RMSE (Test) |
|-------------------|-----------|-------------|
| Linear Regression | 0.041     | 0.496       |
| Random Forest     | 0.029     | 0.499       |

## 5.3 Q3 – Price Optimization  
- **Approach:** Grid search $0.99–$9.99; revenue = price × predicted installs (RF with elasticity)  
- **Elasticity Feature:** price_vs_global_median  
- **Optimal Price:** $9.99  

**Figure 8.** Revenue vs Price Curve  
![Revenue Curve](visualizations/revenue_curve.png)  

# 6. Insights & Recommendations

- **Downloads:** Reviews & recency matter—prompt reviews, update often  
- **Ratings:** Low explained variance—add sentiment & crash metrics  
- **Pricing:** Data lacks true elasticity—run real price experiments  

# 7. Ethics & Interpretability

- **Fairness:** Popular categories over‑represented; watch for bias  
- **Review Manipulation:** Detect spam/fake reviews  
- **Explainability:** Use SHAP & partial‑dependence plots  
- **Governance:** Human‑in‑the‑loop checks

# 8. Conclusion

This project demonstrates a full predictive analytics pipeline on Google Play Store data, yielding actionable insights and highlighting limitations in modeling price sensitivity. Future work should focus on richer behavioral data, segmented pricing strategies, and deploying these models in production.

---

# Appendix

## A. Key Code Snippets

Below are the essential code excerpts needed to reproduce our pipeline:

### 1. Parsing & Cleaning

```python
import numpy as np
import pandas as pd

def parse_reviews(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower().endswith('m'):
        return float(s[:-1]) * 1e6
    if s.lower().endswith('k'):
        return float(s[:-1]) * 1e3
    s_clean = s.replace(',', '')
    return float(s_clean) if s_clean.isdigit() else np.nan

df['Reviews'] = df['Reviews'].apply(parse_reviews)
installs_clean = df['Installs'].str.replace(r'[+,]', '', regex=True)
df['Installs'] = installs_clean.astype(int)

df['Price'] = pd.to_numeric(
    df['Price'].str.replace(r'^\$', '', regex=True),
    errors='coerce'
).fillna(0.0)

def parse_size(size):
    if pd.isna(size) or size == 'Varies with device':
        return np.nan
    s = size.strip()
    if s.endswith('M'):
        return float(s[:-1])
    return float(s[:-1]) / 1024

df['Size'] = df['Size'].apply(parse_size)
```

### 2. Model Training & Evaluation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import r2_score, mean_squared_error

X = df[feature_cols]
y = df['log_Installs']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("LR R²:", r2_score(y_test, y_pred_lr),
      "RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
print("RF R²:", r2_score(y_test, y_pred_rf),
      "RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
```

### 3. Price‑Optimization Simulation

```python
import numpy as np
import pandas as pd

results = []
for price in np.arange(0.99, 10.0, 1.0):
    sim = df[df['is_free'] == 0].copy()
    sim['Price'] = price
    sim['price_vs_global_median'] = price / global_median
    log_installs = rf.predict(sim[feature_cols_elastic])
    installs = np.expm1(log_installs)
    revenue = installs * price
    results.append({'Price': price, 'TotalRevenue': revenue.sum()})

results_df = pd.DataFrame(results)
best = results_df.loc[results_df['TotalRevenue'].idxmax()]
print("Optimal Price:", best.Price)
```

---

## B. Additional Figures

For detailed reference, include full‑size versions of these plots (available in the `visualizations/` folder):

1. **EDA Plots**  
   - Figure 1: Distribution of log(Installs)  
   - Figure 2: Top 10 App Categories  
   - Figure 3: Reviews vs. Installs (log–log)  
   - Figure 4: Rating by Top 5 Categories  
   - Figure 5: Feature Correlation Matrix  

2. **Model Diagnostics**  
   - Figure 6: RF: Actual vs. Predicted Installs  
   - Figure 7: Top 10 Feature Importances  

3. **Price Optimization**  
   - Figure 8: Revenue vs. Price Curve  

*Report prepared by Rithanya Chandran — Contact: Rithanya.Chandran@su.suffolk.edu*  
