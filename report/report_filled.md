<!-- Title Page -->
# Predictive Analytics on Google Play Store Apps

**Rithanya Chandran**  
Master of Science in Business Analytics, Suffolk University  
ISOM 825 – Term Project  
May 5, 2025

---

## 1. Introduction & Dataset Description

### 1.1 Background  
The Google Play Store hosts millions of Android applications, with developers continually striving to optimize downloads, ratings, and revenue. Understanding the factors that drive user engagement and satisfaction can inform data-driven product strategies and pricing decisions. This project applies a full predictive analytics workflow—exploratory analysis, feature engineering, modeling, and ethical reflection—on a real-world dataset of Android apps to generate actionable business insights.

### 1.2 Dataset  
- **Source:** Kaggle “lava18/google-play-store-apps”  
- **Link:** https://www.kaggle.com/datasets/lava18/google-play-store-apps  
- **Contents:** ~10,000 apps with columns: App, Category, Rating, Reviews, Size, Installs, Type, Price, Content Rating, Genres, Last Updated  
- **Rationale:** This dataset presents mixed data types and real-world noise (e.g., “Varies with device” sizes, review count suffixes) across diverse app categories, making it an ideal testbed for predictive modeling relevant to marketplace stakeholders.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Data Overview  
- **Total apps:** 9,660 unique records  
- **Categories:** 34  
- **Missing Ratings:** 1,474 (~15.3%)  
- **Free vs. Paid:** 88% free, 12% paid  
- **Rating Statistics:** Mean = 4.18, Median = 4.30  
- **Install Counts:** Range from 0 to >10,000,000; Median ≈ 50,000

### 2.2 Univariate Analysis  
- **Histogram of log(Installs):** Reveals a heavy right skew, with a long tail of high-download apps.  
- **Boxplot of Ratings:** Concentration between 4.0–4.5, with few apps below 3.0.  
- **Bar Chart of Top Categories:** FAMILY (15%), GAME (10%), TOOLS (9%) dominate.

### 2.3 Bivariate Analysis  
- **Scatter:** log(Installs) vs. log(Reviews) (ρ≈0.65).  
- **Boxplot:** Rating by Category shows EDUCATION and HEALTH categories having higher median ratings.  
- **Heatmap:** Moderate positive correlations among Reviews, Installs, and Size.

### 2.4 Text & Genre Insights  
- **Top Genres:** Tools, Education, Entertainment lead in count.  
- **Observation:** Niche genres like Puzzle and Card see high per-app installs despite lower counts.

### 2.5 Key Takeaways  
1. **`log_Reviews` and `log_Installs`** are strongly correlated (ρ ≈ 0.94).  
2. **Free apps** significantly outperform paid in download volume (88% free).  
3. **Recency** (`days_since_update`) has a moderate positive effect on installs.  
4. **Category and Genre** provide marginal lifts that can segment marketing strategies.

---

## 3. Data Cleaning & Preprocessing

### 3.1 Parsing & Conversion  
- **Reviews:** Converted “M”/“k” suffixes to numeric.  
- **Installs:** Stripped “+” and commas.  
- **Price:** Removed “$”, imputed NAs as \$0.  
- **Size:** Normalized “M”/“k” to MB, set “Varies with device” as missing.  
- **Last Updated:** Parsed into datetime.

### 3.2 Missing Values & Duplicates  
- Dropped duplicates on (App, Category, Last Updated).  
- Imputed Rating NAs with the median (4.30).

### 3.3 Feature Engineering  
- **Skew Reduction:** log_Installs, log_Reviews  
- **Flags:** is_free binary  
- **Recency:** days_since_update  
- **Dummies:** Category, Primary Genre  
- **Elasticity Signal:** price_vs_global_median

---

## 4. Business Questions

1. **Download-Volume Drivers:**  
   _Which app attributes most influence install counts?_  
2. **Rating Prediction:**  
   _Can we predict average user rating from app features?_  
3. **Price Optimization:**  
   _What price maximizes total predicted revenue given price sensitivity?_

---

## 5. Predictive Modeling

### 5.1 Q1 – Installs Prediction  
- **Models:** Linear Regression, Random Forest  
- **Split:** 80% train / 20% test  
- **Results:**  
  | Model             | R² (Test) | RMSE (Test) |  
  |-------------------|-----------|-------------|  
  | Linear Regression | 0.93      | 1.15        |  
  | Random Forest     | 0.93      | 1.14        |  

### 5.2 Q2 – Rating Prediction  
- **Models:** Linear Regression, Random Forest  
- **Results:**  
  | Model             | R² (Test) | RMSE (Test) |  
  |-------------------|-----------|-------------|  
  | Linear Regression | 0.041     | 0.496       |  
  | Random Forest     | 0.029     | 0.499       |  

### 5.3 Q3 – Price Optimization  
- **Approach:** Grid search \$0.99–\$9.99; revenue = price × predicted installs via Random Forest with elasticity.  
- **Elasticity Feature:** price_vs_global_median  
- **Optimal Price:** \$9.99  
- **Revenue Curve:** See figure in Appendix.

---

## 6. Insights & Recommendations

### 6.1 Q1 Insights  
- **Reviews** and **recency** are dominant drivers of downloads.  
- Recommend in-app review prompts and regular updates.

### 6.2 Q2 Insights  
- Low explained variance (R²≈0.03–0.04) suggests missing behavioral features.  
- Suggest sentiment analysis on reviews and session analytics.

### 6.3 Q3 Insights & Limitations  
- Model still treated price neutrally due to limited historical variability.  
- Limitation: elasticity derived from global median only.  
- Recommend collecting time-series price-install data and segment-based pricing.

---

## 7. Ethics & Interpretability

- **Bias & Fairness:** Popular categories over-represented.  
- **Review Manipulation:** High weight on reviews can incentivize spam.  
- **Interpretability:** Use SHAP values and partial-dependence plots.  
- **Governance:** Human oversight on automated pricing strategies.

---

## 8. Conclusion

This project demonstrated an end-to-end predictive analytics workflow on the Play Store dataset. Key findings emphasize the importance of user engagement signals and the need for richer price-sensitivity data. Future work will focus on operationalizing the pipeline and integrating dynamic pricing experiments.

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
