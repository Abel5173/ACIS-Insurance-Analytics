# ACIS Insurance Analytics

## Project Overview
This repository contains the code, analysis, and deliverables for the B5W3: End-to-End Insurance Risk Analytics & Predictive Modeling project, undertaken as part of the TenX Platform challenge. The project is designed for AlphaCare Insurance Solutions (ACIS), a company focused on developing advanced risk and predictive analytics for car insurance planning and marketing in South Africa. The primary goal is to analyze historical insurance claim data (Feb 2014–Aug 2015) to optimize marketing strategies and identify low-risk customer segments for tailored premium pricing, thereby attracting new clients.

The project encompasses:
- Exploratory Data Analysis (EDA)
- A/B Hypothesis Testing
- Statistical Modeling
- Machine Learning

These components work together to derive actionable insights and build predictive models for risk-based pricing. The deliverables aim to provide data-driven recommendations to enhance ACIS's insurance product offerings and marketing strategies.

## Project Purpose
The purpose of this project is to:

1. **Analyze Historical Data**: Explore and summarize insurance claim data to uncover patterns in risk, profitability, and customer behavior.
2. **Validate Hypotheses**: Use A/B hypothesis testing to determine if risk and profit margins differ across provinces, zip codes, and genders.
3. **Build Predictive Models**: Develop machine learning models to predict claim severity and optimize premium pricing based on car, owner, and location features.
4. **Support Business Strategy**: Provide actionable recommendations to tailor insurance products, reduce premiums for low-risk segments, and improve marketing effectiveness.

This project sharpens skills in:
- Data Engineering (DE)
- Predictive Analytics (PA)
- Machine Learning Engineering (MLE)

While emphasizing modular Python code, data versioning, and statistical rigor.

## Repository Structure
```
ACIS-Insurance-Analytics/
│
├── data/
│   ├── raw/                # Raw dataset (tracked with DVC)
│   └── processed/          # Processed data after cleaning and feature engineering
│
├── notebooks/
│   ├── task-1-eda.ipynb    # Exploratory Data Analysis (EDA) and visualizations
│   ├── task-3-hypothesis-testing.ipynb  # A/B hypothesis testing for risk and margin
│   └── task-4-modeling.ipynb  # Predictive modeling and evaluation
│
├── src/
│   ├── data_processing.py  # Scripts for data cleaning and preprocessing
│   ├── eda.py             # Functions for exploratory data analysis
│   ├── hypothesis_testing.py  # Statistical tests for A/B hypothesis
│   ├── modeling.py        # Machine learning model implementation
│   └── utils.py           # Utility functions for visualization and metrics
│
├── reports/
│   ├── interim_report.md   # Interim report summarizing Task 1 and Task 2
│   └── final_report.md    # Final report in Medium blog post format
│
├── .dvc/                   # DVC configuration for data versioning
├── .gitignore              # Gitignore for Python projects
├── README.md               # Project overview (this file)
└── requirements.txt        # Python dependencies
```

### Branches
- `main`: Primary branch with merged, stable code
- `task-1`: EDA and initial data exploration
- `task-2`: Data Version Control (DVC) setup
- `task-3`: A/B hypothesis testing
- `task-4`: Predictive modeling and evaluation

## Key Deliverables

### Task 1: Git and GitHub + Exploratory Data Analysis (EDA)

#### Git Setup
- Initialize a Git repository with a descriptive README
- Implement version control with at least three daily commits and descriptive messages
- Set up CI/CD using GitHub Actions

#### EDA
- Summarize numerical features (e.g., TotalPremium, TotalClaims) with descriptive statistics
- Assess data quality (e.g., missing values, data types)
- Perform univariate analysis (histograms, bar charts) and bivariate/multivariate analysis (scatter plots, correlation matrices)
- Detect outliers using box plots
- Create 3 creative visualizations capturing key insights

Key questions to answer:
- What is the overall loss ratio (TotalClaims / TotalPremium) and its variation by Province, VehicleType, and Gender?
- Are there temporal trends in claim frequency or severity?
- Which vehicle makes/models have the highest/lowest claim amounts?

### Task 2: Data Version Control (DVC)
- Initialize DVC for reproducible data pipelines
- Configure local remote storage and track datasets using DVC
- Commit .dvc files to Git for versioning
- Push data to local remote storage
- Merge task-1 into main via a Pull Request (PR) and create a task-2 branch

### Task 3: A/B Hypothesis Testing
Test the following null hypotheses using statistical tests:
- No risk differences across provinces
- No risk differences between zip codes
- No significant margin (profit) difference between zip codes
- No significant risk difference between women and men

### Task 4: Predictive Modeling
1. **Claim Severity Prediction**:
   - Build models using Linear Regression, Random Forests, and XGBoost
   - Evaluate using RMSE and R-squared

2. **Premium Optimization**:
   - Develop models to predict optimal premiums
   - Advanced: Build binary classification model for claim probability

## Installation and Setup

### Clone the Repository
```bash
git clone https://github.com/Abel5173/ACIS-Insurance-Analytics.git
cd ACIS-Insurance-Analytics
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up DVC
```bash
dvc init
dvc remote add -d localstorage /path/to/your/local/storage
dvc add data/raw/<data.csv>
dvc push
```

## Dependencies
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- dvc, shap, lime

See `requirements.txt` for a complete list.

### Resources
- Insurance Glossary
- DVC Documentation
- Statistical Modeling Resources
- A/B Testing Guide

