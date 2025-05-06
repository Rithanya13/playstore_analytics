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
   ![Distribution of log(Installs)](visualizations/Distribution%20of%20log%28Installs%29.png)

2. **Top 10 App Categories**  
   ![Top 10 App Categories](visualizations/Top%2010%20App%20Categories.png)

3. **Reviews vs. Installs**  
   ![Reviews vs Installs](visualizations/Reviews%20vs.%20Installs.png)

4. **Rating by Top 5 Categories (boxplot)**  
   ![Rating by Top 5 Categories](visualizations/Rating%20by%20Top%205%20Categories%20%28boxplot%29.png)

5. **Feature Correlation Matrix (heatmap)**  
   ![Feature Correlation Matrix](visualizations/Feature%20Correlation%20Matrix%20%28heatmap%29.png)

6. **RF: Actual vs Predicted Installs**  
   ![Actual vs Predicted Installs](visualizations/Actual%20vs%20Predicted%20Installs%20.png)

7. **Top 10 Feature Importances (bar chart)**  
   ![Top 10 Feature Importances](visualizations/Top%2010%20Feature%20Importances%20%28bar%20chart%29.png)

8. **Revenue vs. Price Curve**  
   ![Revenue vs. Price Curve](visualizations/Revenue%20vs.%20Price%20Curve.png)
