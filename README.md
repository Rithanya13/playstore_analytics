# Google Play Store Analytics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/playstore_analysis.ipynb)

## 📄 Project Summary
A predictive analytics project analyzing factors that drive app downloads, user ratings, and pricing strategies on the Google Play Store.

## 🎯 Objectives
- Perform data preprocessing and feature engineering  
- Train and evaluate predictive models for installs and ratings  
- Simulate price optimization for revenue maximization  
- Communicate results through visualizations and written insights  
- Reflect on ethical implications and model interpretability  

## ❓ Business Questions
1. **Download‑Volume Drivers**  
   _Which app attributes most influence install counts?_  
2. **Rating Prediction**  
   _Can we predict an app’s average user rating from its features?_  
3. **Price Optimization**  
   _What price maximizes total predicted revenue, accounting for price sensitivity?_  

## 📂 Dataset
- **Source:** Kaggle “lava18/google-play-store-apps”  
- **Link:** https://www.kaggle.com/datasets/lava18/google-play-store-apps  

## 🛠️ Tools & Libraries
- Python: pandas, NumPy, scikit-learn, matplotlib, seaborn  
- Google Colab for analysis and visualization  
- GitHub for code and report hosting  

## 🚀 How to Run
1. Open the Colab notebook:  
   `notebooks/playstore_analysis.ipynb`  
2. Install dependencies:  
   ```bash
   !pip install pandas numpy scikit-learn matplotlib seaborn kaggle

## 📊 Visualizations

1. **Distribution of log(Installs)**  
   ![Distribution of log(Installs)](https://github.com/user-attachments/assets/b97da62a-4f17-4853-8f2c-cbfe21503410)


2. **Top 10 App Categories**  
   ![Top 10 App Categories](visualizations/bar_top_categories.png)

3. **Reviews vs Installs (log–log)**  
   ![Reviews vs Installs](visualizations/scatter_reviews_vs_installs.png)

4. **Rating by Top 5 Categories**  
   ![Rating by Category](visualizations/boxplot_rating_by_category.png)

5. **Feature Correlation Matrix**  
   ![Feature Correlation Matrix](visualizations/heatmap_correlations.png)

6. **RF: Actual vs Predicted Installs**  
   ![Actual vs Predicted Installs](visualizations/scatter_installs_actual_vs_predicted.png)

7. **Top 10 Feature Importances (Installs Model)**  
   ![Feature Importances](visualizations/bar_feature_importances_installs.png)

8. **Revenue vs Price Curve**  
   ![Revenue Curve](visualizations/revenue_curve.png)
