📊 Customer Churn Prediction System
A comprehensive machine learning project that predicts customer churn for a telecommunications company using multiple modeling techniques including classification, neural networks, survival analysis, and Bayesian methods.
Show Image
Show Image
Show Image

🎯 Project Overview
Customer churn is a critical business metric for subscription-based companies. This project builds a complete churn prediction system that:

Identifies customers at high risk of churning
Quantifies the business impact of churn
Provides actionable insights for retention strategies
Estimates customer lifetime value (CLV)

Business Impact

26.5% of customers churned in the dataset
$2.8M+ estimated annual revenue loss from churn
4.5x higher churn risk for month-to-month contracts
Retention interventions can achieve 200%+ ROI


📁 Project Structure
churn-prediction-system/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── Telco-Customer-Churn.csv    # Original dataset
│   ├── telco_processed.csv          # Processed dataset
│   ├── X_train.csv                  # Training features
│   ├── X_test.csv                   # Test features
│   ├── y_train.csv                  # Training labels
│   └── y_test.csv                   # Test labels
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # Data cleaning & feature engineering
│   ├── 02_exploratory_analysis.ipynb # EDA & visualizations
│   ├── 03_classification_models.ipynb # ML model comparison
│   ├── 04_neural_network.ipynb      # Deep learning approach
│   ├── 05_survival_analysis.ipynb   # Time-to-event modeling
│   └── 06_bayesian_ab_testing.ipynb # Bayesian & Monte Carlo analysis
│
├── src/
│   └── utils.py                     # Utility functions
│
└── outputs/
    ├── figures/                     # 40 visualization plots
    └── models/                      # Saved model files
        ├── best_model_gb.pkl
        ├── logistic_regression.pkl
        ├── random_forest.pkl
        ├── neural_network.pkl
        └── scaler.pkl

🔬 Techniques & Methods
1. Classification Models
ModelROC-AUCPrecisionRecallF1-ScoreGradient Boosting0.8470.6820.5430.605Random Forest0.8390.6710.5210.587Logistic Regression0.8430.6540.5620.604SVM0.8380.6680.5120.580KNN0.7820.5890.4780.528Decision Tree0.7560.5420.5120.527
2. Neural Network

Multi-layer Perceptron (MLP) with various architectures
Best architecture: (128, 64, 32) hidden layers
Early stopping to prevent overfitting
Threshold optimization for improved recall

3. Survival Analysis

Kaplan-Meier Estimator: Non-parametric survival curves
Cox Proportional Hazards: Identifies risk factors with hazard ratios
Log-Rank Tests: Statistical comparison between groups (p < 0.001)

4. Bayesian & Monte Carlo Methods

Bayesian A/B Testing: Contract type impact on churn
Monte Carlo Simulation: Revenue loss estimation
Customer Lifetime Value (CLV): Risk-based segmentation
Intervention ROI Analysis: Cost-benefit optimization


📈 Key Findings
High-Risk Factors (Increase Churn)
FactorChurn RateHazard RatioMonth-to-month contract42.7%4.5xElectronic check payment45.3%1.8xFiber optic internet41.9%1.6xNo tech support41.6%1.4xSenior citizens41.7%1.3x
Protective Factors (Reduce Churn)
FactorChurn RateHazard RatioTwo-year contract2.8%0.2xOne-year contract11.3%0.5xLong tenure (49-72 mo)6.6%0.3xHas dependents15.5%0.8x
Customer Lifetime Value by Segment
SegmentMean CLVMedian CLVLow Risk (Two-year)$2,500+$2,400Medium Risk (One-year)$1,200$1,100High Risk (Month-to-month)$600$500

📊 Visualizations
The project generates 40 publication-ready visualizations:
Sample Outputs
Churn Distribution

Overall churn rate: 26.5%
Class imbalance addressed with stratified sampling

Survival Curves by Contract Type

Month-to-month: Steep decline in first 12 months
Two-year: >95% retention throughout

ROC Curves - Model Comparison

All models significantly outperform random baseline
Gradient Boosting achieves highest AUC

Feature Importance

Top predictors: tenure, contract type, monthly charges
Interpretable coefficients from logistic regression


🛠️ Installation & Usage
Prerequisites
bashPython 3.8+
pip (package installer)
Setup
bash# Clone the repository
git clone https://github.com/yourusername/churn-prediction-system.git
cd churn-prediction-system

# Install dependencies
pip install -r requirements.txt
Requirements
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
lifelines>=0.27.0
joblib>=1.1.0
Running the Notebooks
bash# Start Jupyter
jupyter notebook

# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

💡 Business Recommendations
Based on the analysis, here are actionable recommendations:
1. Contract Strategy

Incentivize longer contracts - Offer discounts for 1-year or 2-year commitments. The churn rate drops from 42.7% to 2.8% with two-year contracts.

2. Payment Method

Encourage automatic payments - Electronic check users have 45% churn vs 15% for credit card auto-pay. Offer incentives to switch.

3. Early Intervention

Target month 1-12 customers - Survival curves show highest churn risk in the first year. Implement onboarding programs.

4. Service Bundling

Promote tech support & security add-ons - Customers without these services have 40%+ churn rates.

5. Senior Customer Program

Create senior-specific retention plans - 41.7% churn rate among seniors requires tailored engagement.


🧠 Skills Demonstrated
CategorySkillsMachine LearningClassification, Ensemble Methods, Neural NetworksStatistical AnalysisSurvival Analysis, Hypothesis Testing, Bayesian InferenceData EngineeringFeature Engineering, Data Preprocessing, Pipeline BuildingBusiness AnalyticsCLV Modeling, ROI Analysis, Monte Carlo SimulationVisualizationMatplotlib, Seaborn, Publication-ready PlotsPythonPandas, NumPy, Scikit-learn, Lifelines

📚 Dataset
Telco Customer Churn Dataset

Source: Kaggle
Size: 7,043 customers, 21 features
Target: Churn (Yes/No)

Features

Demographics: Gender, SeniorCitizen, Partner, Dependents
Services: PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
Account: Contract, PaymentMethod, MonthlyCharges, TotalCharges, Tenure


🔮 Future Improvements

 Deploy model as REST API using FastAPI
 Build interactive Streamlit dashboard
 Implement real-time scoring pipeline
 Add SHAP values for model explainability
 Experiment with XGBoost and LightGBM
 Create automated retraining pipeline


👤 Author
Anita Janie Christdoss Chelladurai

LinkedIn: www.linkedin.com/in/anita-janie-christdoss-chelladurai
GitHub: https://github.com/anita2210
Email: anitajanie2202@gmail.com