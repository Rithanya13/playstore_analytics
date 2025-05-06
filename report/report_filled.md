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
1. **Reviews** are strongly predictive of installs.  
2. **Free apps** significantly outperform paid in download volume.  
3. **Recency** (days since last update) has a moderate positive effect on user engagement.  
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

## Appendix

### A. Code Snippets  
- Parsing & cleaning functions  
- Model training and evaluation calls  
- Price-optimization simulation code

### B. Figures  
- EDA plots (histograms, boxplots, heatmap)  
- Model performance charts (actual vs. predicted)  
- Price optimization revenue curve

---

*Report prepared by Rithanya Chandran. Contact: Rithanya.Chandran@su.suffolk.edu*