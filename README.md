# Sleep Disorder Classification Project

## Overview
This project analyzes sleep health and lifestyle data to predict sleep disorders using various machine learning models. The analysis includes data preprocessing, exploratory data analysis, and the implementation of multiple classification models including Random Forest, XGBoost, and Neural Networks.

## Dataset
The dataset (`Sleep_health_and_lifestyle_dataset.csv`) contains various features related to sleep health and lifestyle, including:
- Personal information (Gender, Age)
- Occupation
- Sleep metrics (Duration, Quality)
- Health indicators (Physical Activity, Stress Level, BMI)
- Vital signs (Blood Pressure, Heart Rate)
- Daily activity (Steps)
- Sleep Disorder classification

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Implementation of multiple machine learning models
- Model performance comparison
- Feature importance analysis

## Models Implemented
1. **Random Forest Classifier**
   - Accuracy: 93.81%
   - Precision: 93.81%
   - Recall: 93.81%
   - F1-score: 93.81%

2. **XGBoost Classifier**
   - Accuracy: 96.46%
   - Precision: 96.52%
   - Recall: 96.46%
   - F1-score: 96.44%

3. **Neural Network**
   - Architecture: 12-8-3 neurons
   - Activation functions: ReLU (hidden layers), Sigmoid (output layer)
   - Performance: ~55.56% accuracy

## Key Findings
- XGBoost performed the best among all models with 96.46% accuracy
- SMOTE was used to handle class imbalance
- Most important features were identified through feature importance plots
- The Neural Network model showed lower performance compared to tree-based models

## Dependencies
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow
- seaborn
- plotly
- matplotlib
- imbalanced-learn

## Usage
1. Load and preprocess the data:
```python
import pandas as pd
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
```

2. Train models:
```python
# Random Forest
rf_model = RandomForestClassifier(random_state=17)
rf_model.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(random_state=17)
xgb.fit(X_train_SMOTE, y_train_SMOTE)

# Neural Network
model_NN = Sequential()
model_NN.add(Dense(12, activation='relu'))
model_NN.add(Dense(8, activation='relu'))
model_NN.add(Dense(3, activation='sigmoid'))
```

## Model Optimization
- Grid Search CV was implemented for Random Forest and XGBoost
- Learning curves were analyzed for model performance
- Speed-score trade-off analysis was performed

## Future Improvements
1. Feature engineering to create more informative predictors
2. Hyperparameter tuning for Neural Network
3. Ensemble modeling
4. Cross-validation for more robust evaluation
5. Deployment pipeline setup


## Contact
[karia.dh@northeastern.edu]
