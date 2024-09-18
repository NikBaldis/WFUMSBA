# Final Project Starter

### Models to be Evaluated.

You will train and evaluate different models

- Logistic Regression: Serves as a baseline for performance comparison.
- Random Forest: An ensemble method known for its robustness and ability to handle complex data structures.
- Gradient Boosting Machine (GBM) OR XGBoost: Advanced ensemble techniques known for their predictive power.
- Neural Network: An approximation method known for it’s ability to identify non-linear relationships.
- StackingClassifier or AutoGluon Weighted Ensemble.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -- model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score

# -- pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from google.colab import drive
drive.mount('/content/drive')

"""
## Import Data

"""

loan = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Machine Learning- S3/Project 3- Predict Loan Default/loan_train.csv')
loan.head()

loan.columns

"""#Preprocessing/Feature Engineering"""

loan['loan_status']

"""Percentage of Good vs Default accounts"""

loan['loan_status'].value_counts(normalize=True)

target = 'loan_status'

"""Change Loan_status (target) to binary"""

loan[target] = loan[target].map({'current': 0, 'default': 1})
loan[target].value_counts()

"""Missing Values"""

loan.isna().sum()

"""Change term, interest rate, employment length, and revol_util to int/float

Change Issue_d and address state to obj

Change Term to float
"""

loan['term']=loan['term'].str.replace('months','').astype('float')
loan['term']=loan['term'].fillna(loan['term'].median())
loan['term']

"""Change interest rate to float"""

loan['int_rate']=loan['int_rate'].str.replace('%','').astype('float')
loan['int_rate']=loan['int_rate'].fillna(loan['int_rate'].median())
loan['int_rate']=loan['int_rate']/100
loan['int_rate']

"""Change Employment Length to int"""

loan['emp_length']=loan['emp_length'].str.replace('years','').str.replace('year','').str.replace('<','').str.replace('+','').astype('float')
loan['emp_length']=loan['emp_length'].fillna(loan['emp_length'].median())
loan['emp_length']

"""Change revol_util to float"""

loan['revol_util']=loan['revol_util'].str.replace('%','').astype('float')
loan['revol_util']=loan['revol_util'].fillna(loan['revol_util'].median())
loan['revol_util']=loan['revol_util']/100
loan['revol_util']

"""Change issue_d to date"""

#loan['issue_d'] = pd.to_datetime(loan['issue_d'], format="%b-%Y")
#loan['issue_d']

"""Change address state to object"""

loan['addr_state']=loan['addr_state'].fillna('missing')
loan['addr_state']

loan['loan_amnt']

"""Find Numeric and Categorical Features"""

numeric_features = loan.select_dtypes(include=['int64', 'float64']).columns
#print(numeric_features)
numeric_features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'total_acc', 'out_prncp', 'out_prncp_inv', 'total_rec_late_fee',
       'last_pymnt_amnt', 'delinq_amnt',
       'pub_rec_bankruptcies', 'tax_liens','int_rate','emp_length']
print(numeric_features)

categorical_features = loan.select_dtypes(include=['object']).columns
#print(categorical_features)
categorical_features = ['grade', 'sub_grade','home_ownership', 'verification_status','issue_d','addr_state','term']
print(categorical_features)

"""#Prepare Pipeline"""

# create transformers

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

"""#Exploratory Analysis

Distribution of Numerical Columns
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")

for col in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data= loan, x=col, hue=target, kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()

"""Insights:
- 5000 is usually the most fraudulent account loan amount
- 4000 is the most commonly invested amount for the fraudulent loans
- Debt to Income is pretty consistent for fraudulent accounts between about 13% to 21%, significantly drops off before about 5% and after 25%
"""

# Compute the correlation matrix
corr = loan[numeric_features].corr()

# Generate a heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix of Numeric Features')
plt.show()

"""Insights:
- Months since last record and number of derogatory public records are highly correlated, also bankrupcies are highly correlated with months since last record

Distributions of Categorical Columns
"""

for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data= loan, x=col, hue=target, kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()

"""Insights:
- C and D grades are the most fraudulent
- C1&2 and B3&5 are the most fraudulent subgrades
- Verification status does not seem to play a big role in fraudulent accounts

# Train Test Split
"""

X_train, X_test, y_train, y_test = train_test_split(loan[numeric_features + categorical_features], loan[target], test_size=0.2, random_state=42)

"""#Logistic Regression"""

# Define the Logistic Regression pipeline
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=0, max_iter=300))])

# Train the Logistic Regression model
lr_pipeline.fit(X_train, y_train)

# Predict and evaluate the model
lr_predictions = lr_pipeline.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_predictions):.4f}")
print(f"  AUC: {roc_auc_score(y_test, lr_predictions):.4f}")
print(f"  Precision: {precision_score(y_test, lr_predictions):.4f}")
print(f"  Recall: {recall_score(y_test, lr_predictions):.4f}")
print(f"  F1: {f1_score(y_test, lr_predictions):.4f}")

feature_names = preprocessor.get_feature_names_out()
feature_importance = lr_pipeline.named_steps['classifier'].coef_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance[0]})
feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

# Logistic Regression coefficients as feature importance
lr_coefficients = lr_pipeline.named_steps['classifier'].coef_[0]

# Aligning feature names and coefficients
lr_feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': lr_coefficients})
lr_feature_importance_df = lr_feature_importance_df.sort_values(by='Coefficient', ascending=False)
lr_feature_importance_df.head(10)

"""Plot ROC and PR Curves"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

lr_predictions_proba = lr_pipeline.predict_proba(X_test)[:,1]

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, lr_predictions)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, lr_predictions)
pr_auc = auc(recall, precision)

# Plot ROC Curve
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Highlighting the 5% FPR point
idx = next(i for i, x in enumerate(fpr) if x >= 0.05)  # Find the index of the FPR just over 5%
plt.plot(fpr[idx], tpr[idx], 'ro', label='~5% FPR')  # 'ro' for red dot
plt.annotate(f'FPR ~5%\nTPR={tpr[idx]:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(10,-10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

"""#Random Forest"""

from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=50, n_jobs = -1, random_state=0))])

# Train the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Predict and evaluate the model
rf_predictions = rf_pipeline.predict(X_test)
rf_predictions_proba = rf_pipeline.predict_proba(X_test)[:,1]
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(f"  AUC: {roc_auc_score(y_test, rf_predictions):.4f}")
print(f"  Precision: {precision_score(y_test, rf_predictions):.4f}")
print(f"  Recall: {recall_score(y_test, rf_predictions):.4f}")
print(f"  F1: {f1_score(y_test, rf_predictions):.4f}")

from sklearn.model_selection import GridSearchCV
# Create the parameter grid
param_grid = {
    'classifier__n_estimators': [20, 30, 50],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [ 5, 10],
    #'classifier__min_samples_leaf': [1, 2, 4]
}

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best set of hyperparameters
best_params = grid_search.best_params_

# Print the best set of hyperparameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Initialize the pipeline with the preprocessor and a Random Forest classifier
rf_pipeline_hpo = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=50,
                                                                    min_samples_split=10,
                                                                    max_depth = None,
                                                                    n_jobs=-1,
                                                                    random_state=42))])
# Train the pipeline
rf_pipeline_hpo.fit(X_train, y_train)

# compare rf_pipeline to rf_pipeline_hpo performance
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
y_pred_rf_hpo = rf_pipeline_hpo.predict(X_test)
y_pred_proba_rf_hpo = rf_pipeline_hpo.predict_proba(X_test)[:, 1]

# Evaluation Metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)


# Evaluation Metrics
#print("Random Forest Model Evaluation:")
#print(f"  Accuracy: {accuracy_rf:.4f}")
#print(f"  AUC: {auc_rf:.4f}")
#print(f"  Precision: {precision_rf:.4f}")
#print(f"  Recall: {recall_rf:.4f}")
#print(f"  F1: {f1_score(y_test, y_pred_rf):.4f}")
#print("\n -------- \n")
print("Random Forest Model Evaluation with HPO:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf_hpo):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_rf_hpo):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf_hpo):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_rf_hpo):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_rf_hpo):.4f}")

"""#**False Positive Rate for RF Question - 2%- Threshold 0.464092  5%-0.365205**"""

from sklearn.metrics import roc_curve
import numpy as np
# Predict probabilities for the positive class
y_scores = rf_pipeline_hpo.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# Define target FPR values
target_fpr = np.arange(0.01, 0.11, 0.01)  # From 1% to 10%

# Interpolate to find TPR and threshold for target FPRs
interp_tpr = np.interp(target_fpr, fpr, tpr)
interp_thresholds = np.interp(target_fpr, fpr, thresholds)

# Print the results
for i in range(len(target_fpr)):
    print(f"Target FPR: {target_fpr[i]:.2f}, Expected TPR: {interp_tpr[i]:.4f}, Threshold: {interp_thresholds[i]:.4f}")

import pandas as pd

# Create a DataFrame from the target FPR, interpolated TPR, and interpolated thresholds
target_fpr_df = pd.DataFrame({
    'Target FPR (%)': target_fpr * 100,  # Convert to percentage
    'Expected TPR': interp_tpr,
    'Threshold': interp_thresholds
})

# Display the DataFrame
target_fpr_df

# Adjusting the feature name extraction for OneHotEncoder to use get_feature_names_out
feature_names = list(preprocessor.transformers_[0][2]) + \
    list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))

rf_importances = rf_pipeline_hpo.named_steps['classifier'].feature_importances_

# Display the top 10 features
rf_feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances})
rf_feature_importance_df = rf_feature_importance_df.sort_values(by='Importance', ascending=False)
rf_feature_importance_df.head(10)

"""Plot ROC and PR Curves"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_rf)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_rf)
pr_auc = auc(recall, precision)

# Plot ROC Curve
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Highlighting the 5% FPR point
idx = next(i for i, x in enumerate(fpr) if x >= 0.05)  # Find the index of the FPR just over 5%
plt.plot(fpr[idx], tpr[idx], 'ro', label='~5% FPR')  # 'ro' for red dot
plt.annotate(f'FPR ~5%\nTPR={tpr[idx]:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(10,-10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

"""#GBM"""

from sklearn.ensemble import GradientBoostingClassifier #- GBM classfier
# Define the GBMClassifier - here we are not using the pipeline just model

gbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(
                                                                        learning_rate=0.1, n_estimators=100,
                                                                    ))])

# Train the GBMClassifier model
gbm_pipeline.fit(X_train,y_train)


y_pred_gbm  = gbm_pipeline.predict(X_test)
y_pred_proba_gbm = gbm_pipeline.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print("\n -------- ")
print("GBM Baseline:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_gbm):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_gbm):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_gbm):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_gbm):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_gbm):.4f}")

"""#Optimize GBM"""

# Create the parameter grid
param_grid = {
    'classifier__n_estimators': [100,200],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
}

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(gbm_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best set of hyperparameters
best_params = grid_search.best_params_

# Print the best set of hyperparameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Initialize the pipeline with the preprocessor and a GBM classifier
gbm_pipeline_hpo = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(n_estimators=200,
                                                                    learning_rate=0.2,
                                                                    ))])
# Train the pipeline
gbm_pipeline_hpo.fit(X_train, y_train)

# compare gbm_pipeline to gbm_pipeline_hpo performance

y_pred_gbm  = gbm_pipeline.predict(X_test)
y_pred_proba_gbm = gbm_pipeline.predict_proba(X_test)[:, 1]

y_pred_gbm_hpo  = gbm_pipeline_hpo.predict(X_test)
y_pred_proba_gbm_hpo = gbm_pipeline_hpo.predict_proba(X_test)[:, 1]

# Evaluation Metrics

#print("\n -------- ")
#print("GBM Baseline:")
#print(f"  Accuracy: {accuracy_score(y_test, y_pred_gbm):.3f}")
#print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_gbm):.3f}")
#print(f"  Precision: {precision_score(y_test, y_pred_gbm):.3f}")
#print(f"  Recall: {recall_score(y_test, y_pred_gbm):.4f}")
#print(f"  F1: {f1_score(y_test, y_pred_gbm):.4f}")

print("\n -------- ")
print("GBM Optimized:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_gbm_hpo):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_gbm_hpo):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_gbm_hpo):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_gbm_hpo):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_gbm_hpo):.4f}")

"""#**False Positive Rate for GBM Question - 2%- 0.543357  5%- 0.389132**"""

from sklearn.metrics import roc_curve
import numpy as np
# Predict probabilities for the positive class
y_scores = gbm_pipeline_hpo.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# Define target FPR values
target_fpr = np.arange(0.01, 0.11, 0.01)  # From 1% to 10%

# Interpolate to find TPR and threshold for target FPRs
interp_tpr = np.interp(target_fpr, fpr, tpr)
interp_thresholds = np.interp(target_fpr, fpr, thresholds)

# Print the results
for i in range(len(target_fpr)):
    print(f"Target FPR: {target_fpr[i]:.2f}, Expected TPR: {interp_tpr[i]:.4f}, Threshold: {interp_thresholds[i]:.4f}")

import pandas as pd

# Create a DataFrame from the target FPR, interpolated TPR, and interpolated thresholds
target_fpr_df = pd.DataFrame({
    'Target FPR (%)': target_fpr * 100,  # Convert to percentage
    'Expected TPR': interp_tpr,
    'Threshold': interp_thresholds
})

# Display the DataFrame
target_fpr_df

# Adjusting the feature name extraction for OneHotEncoder to use get_feature_names_out
feature_names = list(preprocessor.transformers_[0][2]) + \
    list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))

gbm_importances = gbm_pipeline_hpo.named_steps['classifier'].feature_importances_

# Display the top 10 features
gbm_feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': gbm_importances})
gbm_feature_importance_df = gbm_feature_importance_df.sort_values(by='Importance', ascending=False)
gbm_feature_importance_df.head(10)

"""Plot ROC and PR Curves"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_gbm_hpo)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_gbm_hpo)
pr_auc = auc(recall, precision)

# Plot ROC Curve
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Highlighting the 5% FPR point
idx = next(i for i, x in enumerate(fpr) if x >= 0.05)  # Find the index of the FPR just over 5%
plt.plot(fpr[idx], tpr[idx], 'ro', label='~5% FPR')  # 'ro' for red dot
plt.annotate(f'FPR ~5%\nTPR={tpr[idx]:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(10,-10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

"""#Neural Network"""

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, average_precision_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

categorical_features

#from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

#X = loan.drop('target', axis=1)
#y = loan['target']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#numeric_features = ['loan_amnt','funded_amnt','funded_amnt_inv','installment','annual_inc','dti','delinq_2yrs',
# 'fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec',
# 'revol_bal','total_acc','out_prncp','out_prncp_inv','total_rec_late_fee','last_pymnt_amnt','delinq_amnt','pub_rec_bankruptcies','tax_liens',
# 'int_rate','emp_length']
#categorical_features = ['grade','sub_grade','home_ownership','verification_status','issue_d','addr_state','term']

# Create preprocessing pipelines
#numeric_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='median')),
#    ('scaler', StandardScaler())
#])

#categorical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#    ('onehot', OneHotEncoder(handle_unknown='ignore'))
#])

# Combine preprocessing steps
#preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', numeric_transformer, numeric_features),
#        ('cat', categorical_transformer, categorical_features)
#    ])

# Create the MLPRegressor pipeline
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPClassifier(random_state=42))
])


# Train the model
mlp_pipeline.fit(X_train, y_train)



# Predict and evaluate the model
y_pred_nn = mlp_pipeline.predict(X_test)
y_pred_proba_nn = mlp_pipeline.predict_proba(X_test)[:, 1]
mse = mean_squared_error(y_test, y_pred_nn)
r2 = r2_score(y_test, y_pred_nn)

print("Neural Network:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_nn):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_nn):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_nn):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_nn):.4f}")

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Compute permutation importance
result = permutation_importance(mlp_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)

# Function to get feature names from column transformer
def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    output_features = []

    # Loop through each transformer in the column transformer
    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(pipe, 'get_feature_names_out'):
            # If the transformer has a get_feature_names_out method, use it
            if hasattr(pipe, 'categories_'):
                feature_names = pipe.get_feature_names_out(features)
            else:
                feature_names = pipe.get_feature_names_out()
        else:
            # Otherwise, use the provided feature names
            feature_names = features
        output_features.extend(feature_names)
    return output_features

# Extract feature names from the preprocessor
feature_names = get_feature_names(preprocessor)

# Now using feature_names with sorted_idx for labeling in the plot
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 7))
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(feature_names)[sorted_idx])
plt.title("Permutation Importance (test set)")
plt.tight_layout()
plt.show()

#from sklearn.inspection import PartialDependenceDisplay

#common_params = {
#    "subsample": 200,
#    "n_jobs": -1,
#    "grid_resolution": 40,
#    "random_state": 0,
#}


#print("Computing partial dependence plots...")
#features_info = {
    # features of interest
#    "features": numeric_features,
#    # type of partial dependence plot
#   "kind": "average",
#    # information regarding categorical features
#    "categorical_features": categorical_features,
#}

#_, ax = plt.subplots(ncols=5, nrows=5, figsize=(9, 8), constrained_layout=True)
#display = PartialDependenceDisplay.from_estimator(
#    mlp_pipeline,
#    X_train,
#    **features_info,
#    ax=ax,
#    **common_params,
#)
#_ = display.figure_.suptitle(
#    (
#        "Partial dependence of California Dataset\n"
#        "with an MLPRegressor"
#    ),
#    fontsize=12,
#)

X_pd_nn=loan.drop('loan_status', axis=1)

X_s = loan[numeric_features + categorical_features]
y_s = mlp_pipeline.predict(X_pd_nn)
Z = pd.concat([X_s, pd.Series(y_s, name='pred')], axis=1)

sns.barplot(data=Z,x=categorical_features[0],y='pred')
plt.title("kind of partial dependance for category")
plt.show()

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

X = loan.drop('loan_status', axis=1)

# Split dataset into features and target variable
X_s = loan[numeric_features + categorical_features]
y_s = mlp_pipeline.predict(X)

# Splitting dataset into training and testing sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

# Creating transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    (  'scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating the Linear Regression pipeline
linear_regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fitting the model
linear_regression_pipeline.fit(X_train_s, y_train_s)

# Calculate R^2 score on the training set
sur_pred = linear_regression_pipeline.predict(X_train_s)
r2_score_train = r2_score(y_train_s, sur_pred)
print(f"R^2 score on the training set: {r2_score_train:.4f}")

# Accessing and printing the model's coefficients
regressor = linear_regression_pipeline.named_steps['regressor']
feature_names_transformed = (numeric_features +
                             list(linear_regression_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)))

print("Coefficients:")
for name, coef in zip(feature_names_transformed, regressor.coef_):
    print(f"{name}: {coef:.4f}")


nn_coefficients = pd.DataFrame({
    'Feature': feature_names_transformed,
    'Coefficient': regressor.coef_
})

nn_coefficients['pos_neg'] = nn_coefficients['Coefficient'].apply(lambda x: 'positive' if x > 0 else 'negative')

nn_coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)

# Plotting the coefficients
sns.barplot(data=nn_coefficients, y='Feature', x='Coefficient', hue='pos_neg')
plt.xticks(rotation=90)
plt.title('Coefficients')
plt.show()

nn_feature_importances = nn_coefficients.sort_values(by='Coefficient', ascending=False)
nn_feature_importances

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_nn)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_nn)
pr_auc = auc(recall, precision)

# Plot ROC Curve
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Highlighting the 5% FPR point
idx = next(i for i, x in enumerate(fpr) if x >= 0.05)  # Find the index of the FPR just over 5%
plt.plot(fpr[idx], tpr[idx], 'ro', label='~5% FPR')  # 'ro' for red dot
plt.annotate(f'FPR ~5%\nTPR={tpr[idx]:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(10,-10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

"""#Stacking Classifier

Fit the Pipeline
"""

# base estimators for stacker
base_estimators = [
    ('gbm', GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50,max_depth=None,min_samples_split=10, random_state=42)),
    ('nn', MLPClassifier(random_state=42))
]

# final estimator on top
final_estimator = LogisticRegression()

stacking_classifier = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=3,
    n_jobs=-1
)

stack_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', stacking_classifier)])

stack_pipeline.fit(X_train, y_train)

"""# Evaluate"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Predictions for the training set
y_train_pred = stack_pipeline.predict(X_train)
y_train_prob = stack_pipeline.predict_proba(X_train)[:, 1]

# Predictions for the test set
y_test_pred = stack_pipeline.predict(X_test)
y_test_prob = stack_pipeline.predict_proba(X_test)[:, 1]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Binarize labels for AUC calculation
lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train).ravel()
y_test_binarized = lb.transform(y_test).ravel()

# Calculating metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred, )
test_precision = precision_score(y_test, y_test_pred, )

train_recall = recall_score(y_train, y_train_pred, )
test_recall = recall_score(y_test, y_test_pred, )

train_f1 = f1_score(y_train, y_train_pred, )
test_f1 = f1_score(y_test, y_test_pred, )

train_auc = roc_auc_score(y_train_binarized, y_train_prob)
test_auc = roc_auc_score(y_test_binarized, y_test_prob)

# Print Metrics
print("Training Metrics:")
print(f"Accuracy: {train_accuracy:.2f}")
print(f"Precision (default): {train_precision:.2f}")
print(f"Recall (default): {train_recall:.2f}")
print(f"F1 Score (default): {train_f1:.2f}")
print(f"AUC: {train_auc:.2f}")

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision (default): {test_precision:.2f}")
print(f"Recall (default): {test_recall:.2f}")
print(f"F1 Score (default): {test_f1:.2f}")
print(f"AUC: {test_auc:.2f}")

"""## Permutation Importance

"""

from sklearn.inspection import permutation_importance
result = permutation_importance(stack_pipeline, X_test, y_test,
                                n_repeats=10, random_state=42,
                                n_jobs=-1)

def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    feature_names = []

    # Loop through each transformer within the ColumnTransformer
    for name, transformer, columns in column_transformer.transformers_:
        if name == 'remainder':  # Skip the 'remainder' transformer, if present
            continue
        if isinstance(transformer, Pipeline):
            # If the transformer is a pipeline, get the last transformer from the pipeline
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):
            # If the transformer has 'get_feature_names_out', use it
            names = list(transformer.get_feature_names_out(columns))
        else:
            # Otherwise, just use the column names directly
            names = list(columns)

        feature_names.extend(names)

    return feature_names

transformed_feature_names = get_feature_names(preprocessor)
transformed_feature_names

feature_names = numeric_features + categorical_features

for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
        print(f"Feature {feature_names[i]} "
              f"Mean Importance: {result.importances_mean[i]:.3f} "
              f"+/- {result.importances_std[i]:.3f}")

stack_feature_importances_df = pd.DataFrame({
  'Feature': feature_names,  # Or 'feature_names' if applicable
  'Importance Mean': result.importances_mean,
  'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False).reset_index(drop=True)
stack_feature_importances_df

plt.figure(figsize=(10, 6))
sns.barplot(stack_feature_importances_df, x='Importance Mean', y='Feature')
plt.title('Permutation Importance')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test_binarized, y_test_prob)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test_binarized, y_test_prob)
pr_auc = auc(recall, precision)

# Plot ROC Curve
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Highlighting the 5% FPR point
idx = next(i for i, x in enumerate(fpr) if x >= 0.05)  # Find the index of the FPR just over 5%
plt.plot(fpr[idx], tpr[idx], 'ro', label='~5% FPR')  # 'ro' for red dot
plt.annotate(f'FPR ~5%\nTPR={tpr[idx]:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(10,-10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Assuming calculations for fpr, tpr, and thresholds_roc are already done

plt.figure(figsize=(14, 6))

# Plot ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Highlight the 5% FPR with a vertical line
idx = next(i for i, x in enumerate(fpr) if x >= 0.05)  # Find the index for FPR just over 5%
plt.axvline(x=fpr[idx], color='r', linestyle='--')  # Vertical line for ~5% FPR
plt.plot(fpr[idx], tpr[idx], 'ro')  # Red dot at the intersection

# Adding a text annotation for the threshold
plt.annotate(f'Threshold={thresholds_roc[idx]:.2f}\nTPR/Recall={tpr[idx]:.2f}\n FPR = 5%', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(-10,10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

import numpy as np

# Find the closest threshold in the PR curve to the one identified in the ROC curve analysis
# This might not be exact due to the different metrics, but we find the nearest one
roc_threshold = thresholds_roc[idx]
closest_threshold_index = np.argmin(np.abs(thresholds_pr - roc_threshold))
selected_precision = precision[closest_threshold_index]
selected_recall = recall[closest_threshold_index]

plt.figure(figsize=(7, 5))

# Plot PR Curve
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Highlight the selected threshold
plt.plot(selected_recall, selected_precision, 'ro')  # Red dot at the selected threshold
plt.annotate(f'Threshold={roc_threshold:.2f}\nPrecision={selected_precision:.2f}\nRecall={selected_recall:.2f}',
             (selected_recall, selected_precision),
             textcoords="offset points",
             xytext=(-10,10),
             ha='center')

plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

"""#Feature Importance Plots for Optimized Models"""

import matplotlib.pyplot as plt

# Plot for Logistic Regression
plt.figure(figsize=(10, 6))
plt.title('Top 10 Features Coefficients in Logistic Regression')
plt.barh(lr_feature_importance_df['Feature'][:10], lr_feature_importance_df['Coefficient'][:10])
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# Plot for Optimized Random Forest
plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances in Optimized Random Forest')
plt.barh(rf_feature_importance_df['Feature'][:10], rf_feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# Plot for Optimized GBM
plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances in Optimized GBM Forest')
plt.barh(gbm_feature_importance_df['Feature'][:10], gbm_feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

#Plot for Neaural Network
plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances in Neural Network')
plt.barh(nn_feature_importances['Feature'][:10], nn_feature_importances['Coefficient'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

#Plot for Stacking
plt.figure(figsize=(10, 6))
plt.title('Top 10 Feature Importances in Stacking Classifier')
plt.barh(stack_feature_importances_df['Feature'][:10], stack_feature_importances_df['Importance Mean'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

"""#Model Curves for Optimized Models"""

model_set = {
    "LR": lr_pipeline,
    "RF": rf_pipeline_hpo,
    "GBM": gbm_pipeline_hpo,
    "Neural Network": mlp_pipeline,
    "Stacking Classifier": stack_pipeline}


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

def plot_model_curves(model_set, X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(loan[numeric_features + categorical_features], loan[target], test_size=0.2, random_state=42)

    plt.figure(figsize=(15, 7))

    # ROC Curve plot
    plt.subplot(1, 2, 1)
    for name, model in model_set.items():
        # Fit the model
        model.fit(X_train, y_train)
        # Get predicted probabilities
        y_score = model.predict_proba(X_test)[:, 1]
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        # Plot
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # PR Curve plot
    plt.subplot(1, 2, 2)
    for name, model in model_set.items():
        # Get predicted probabilities
        y_score = model.predict_proba(X_test)[:, 1]
        # Compute PR curve and PR area
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)
        # Plot
        plt.plot(recall, precision, label=f'{name} (average precision = {average_precision:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# Example usage:
model_set = {
    "LR": lr_pipeline,
    "RF": rf_pipeline_hpo,
    "GBM": gbm_pipeline_hpo,
    "Neural Network": mlp_pipeline,
    "Stacking Classifier": stack_pipeline}

#plot_model_curves(model_set, test[numeric_features], test['riskperformance_target'])

plot_model_curves(model_set, X_test, y_test)

"""#Which is Best Model?"""

#paste eval metrics of lr, optimized rf and gbm

print(f"Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions):.4f}")
print(f"  AUC: {roc_auc_score(y_test, lr_predictions):.4f}")
print(f"  Precision: {precision_score(y_test, lr_predictions):.4f}")
print(f"  Recall: {recall_score(y_test, lr_predictions):.4f}")
print(f"  F1: {f1_score(y_test, lr_predictions):.4f}")
print("\n -------- \n")
print("Random Forest Model Optimized:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf_hpo):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_rf_hpo):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf_hpo):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_rf_hpo):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_rf_hpo):.4f}")
print("\n -------- ")
print("GBM Optimized:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_gbm_hpo):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_gbm_hpo):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_gbm_hpo):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_gbm_hpo):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_gbm_hpo):.4f}")
print("\n -------- ")
print("Neural Network:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba_nn):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_nn):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_nn):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_nn):.4f}")
print("\n -------- ")
print("Stacking Classifier:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"AUC: {test_auc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

"""#Best Model Partial Dependance Plot"""

# !pip install -U scikit-learn
numeric_features

X_train.columns

var = 'loan_amnt'
sample_n = 1000
pdp_values = pd.DataFrame(X_train[var].sort_values().sample(frac=0.2).unique(),columns=[var])
pdp_sample = X_train.sample(sample_n).drop(var, axis=1)

pdp_cross = pdp_sample.merge(pdp_values, how='cross')
pdp_cross['pred'] = gbm_pipeline_hpo.predict_proba(pdp_cross)[:,1]
plt.figure(figsize=(10, 3))
sns.lineplot(x=f"{var}", y='pred', data=pdp_cross)
plt.title(f"Partial Dependance Plot: {var}")
plt.ylabel('Predicted Probability')
plt.xticks(rotation=45)
plt.ylim(0.0, 1)
plt.grid(True)
plt.show()

def pdp_plot_numeric(var, sample_n):
  # var = 'credit_amount'
  pdp_values = pd.DataFrame(X_train[var].sort_values().sample(frac=0.1).unique(),columns=[var])
  pdp_sample = X_train.sample(sample_n).drop(var, axis=1)

  pdp_cross = pdp_sample.merge(pdp_values, how='cross')
  pdp_cross['pred'] = gbm_pipeline_hpo.predict_proba(pdp_cross)[:,1]
  plt.figure(figsize=(10, 3))
  sns.lineplot(x=f"{var}", y='pred', data=pdp_cross)
  plt.title(f"Partial Dependance Plot: {var}")
  plt.ylabel('Predicted Probability')
  plt.xticks(rotation=45)
  #plt.ylim(0, 1)
  plt.grid(True)
  plt.show()

# numeric_features = ['credit_amount', 'duration', 'age']
for var in numeric_features:
  pdp_plot_numeric(var, sample_n=300)

"""# PDP Categorical"""

def pdp_plot_categorical(var, sample_n):
  sns.set_style("whitegrid")  # Try "darkgrid", "ticks", etc.
  sns.set_context("notebook")  # Try "paper", "notebook", "poster" for different sizes

  pdp_values = pd.DataFrame(X_test[var].sort_values().unique(),columns=[var])
  pdp_sample = X_test.sample(sample_n).drop(var, axis=1)

  pdp_cross = pdp_sample.merge(pdp_values, how='cross')
  pdp_cross['pred'] = gbm_pipeline_hpo.predict_proba(pdp_cross)[:,1]
  mean_pred = pdp_cross['pred'].mean()
  pdp_cross['pred'] = pdp_cross['pred'].apply(lambda x: x - mean_pred)
  plt.figure(figsize=(10, 3))
 #sns.lineplot(x=f"{var}", y='pred', data=pdp_cross)
  sns.barplot(x=f"{var}", y='pred',
              ci=None,
              data=pdp_cross,
              estimator="mean")
  plt.title(f"Partial Dependance Plot: {var}")
  plt.ylabel('Predicted Probability')
  plt.xticks(rotation=45)
  #plt.ylim(0, 1)
  plt.grid(True)
  plt.show()


for var in categorical_features:
  pdp_plot_categorical(var, sample_n=100)

!pip install dalex

import dalex as dx # for explanations
pipeline_explainer = dx.Explainer(gbm_pipeline_hpo, X_test, y_test)
pipeline_explainer

model_performance  = pipeline_explainer.model_performance("classification")
model_performance.result

"""# Variable Importance"""

# Calculate feature importance
fi = pipeline_explainer.model_parts(processes=4)

# Plot feature importance
fi.plot()

"""## PDP"""

# Let's say you want to create PDPs for a feature named 'feature_name'
pdp_numeric_profile = pipeline_explainer.model_profile(variables=numeric_features)

# Now, plot the PDP for 'feature_name'
pdp_numeric_profile.plot()

pdp_categorical_profile = pipeline_explainer.model_profile(
    variable_type = 'categorical',
    variables=categorical_features)

# Now, plot the PDP for 'feature_name'
pdp_categorical_profile.plot()

"""# Local predictions"""

X_test['pred']= gbm_pipeline_hpo.predict(X_test)
X_test['pred_proba']= gbm_pipeline_hpo.predict_proba(X_test)[:,1]
X_test[target] = y_test
X_test.head()

X_test['loan_status'].head()

"""#True Positives"""

top_10_tp = (X_test
             .query('loan_status == 1 and pred == 1')
             .sort_values(by='pred_proba', ascending=False)
             .head(10)
             .reset_index(drop=True)
)
top_10_tp

"""# Shap Explainations TP"""

for index, row in top_10_tp.iterrows():
  local_breakdown_exp = pipeline_explainer.predict_parts(
      top_10_tp.iloc[index],
      type='shap',
      B=5,
      label=f"record:{index}, prob:{row['pred_proba']:.3f}")

  local_breakdown_exp.plot()

"""# Break Down Interactions TP"""

for index, row in top_10_tp.iterrows():
  local_breakdown_exp = pipeline_explainer.predict_parts(
      top_10_tp.iloc[index],
      type='break_down_interactions',
      label=f"record:{index}, prob:{row['pred_proba']:.3f}")

  local_breakdown_exp.plot()

"""# False Positives"""

top_10_fp = (X_test
             .query('loan_status == 0 and pred == 1')
             .sort_values(by='pred_proba', ascending=False)
             .head(10)
             .reset_index(drop=True)
)
top_10_fp

"""# Shap Explainations FP"""

for index, row in top_10_fp.iterrows():
  local_breakdown_exp = pipeline_explainer.predict_parts(
      top_10_fp.iloc[index],
      type='shap',
      label=f"record:{index}, prob:{row['pred_proba']:.3f}")

  local_breakdown_exp.plot()

"""# Break Down Interactions FP"""

for index, row in top_10_fp.iterrows():
  local_breakdown_exp = pipeline_explainer.predict_parts(
      top_10_fp.iloc[index],
      type='break_down_interactions',
      label=f"record:{index}, prob:{row['pred_proba']:.3f}")

  local_breakdown_exp.plot()

"""# False Negatives
---
"""

top_10_fn = (X_test
             .query('loan_status == 0 and pred == 1')
             .sort_values(by='pred_proba', ascending=True)
             .head(10)
             .reset_index(drop=True)
)
top_10_fn

"""#Shap FN

"""

for index, row in top_10_fn.iterrows():
  local_breakdown_exp = pipeline_explainer.predict_parts(
      top_10_fn.iloc[index],
      type='shap',
      B=5,
      label=f"record:{index}, prob:{row['pred_proba']:.3f}")

  local_breakdown_exp.plot()

"""# Break Down Interactions FN"""

for index, row in top_10_fn.iterrows():
  local_breakdown_exp = pipeline_explainer.predict_parts(
      top_10_fn.iloc[index],
      type='break_down_interactions',
      label=f"record:{index}, prob:{row['pred_proba']:.3f}")

  local_breakdown_exp.plot()

"""#Apply best model to holdout set"""

# Load the dataset
df2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Machine Learning- S3/Project 3- Predict Loan Default/loan_holdout.csv')

# Display the first few rows of the dataframe
df2.head()

target='loan_status'

"""#Clean

Missing Values
"""

df2.isna().sum()

"""Change Term to float"""

df2['term']=df2['term'].str.replace('months','').astype('float')
df2['term']=df2['term'].fillna(df2['term'].median())
df2['term']

"""Change interest rate to float"""

df2['int_rate']=df2['int_rate'].str.replace('%','').astype('float')
df2['int_rate']=df2['int_rate'].fillna(loan['int_rate'].median())
df2['int_rate']=df2['int_rate']/100
df2['int_rate']

"""Change Employment Length to int"""

df2['emp_length']=df2['emp_length'].str.replace('years','').str.replace('year','').str.replace('<','').str.replace('+','').astype('float')
df2['emp_length']=df2['emp_length'].fillna(df2['emp_length'].median())
df2['emp_length']

"""Change revol_util to float"""

df2['revol_util']=df2['revol_util'].str.replace('%','').astype('float')
df2['revol_util']=df2['revol_util'].fillna(loan['revol_util'].median())
df2['revol_util']=df2['revol_util']/100
df2['revol_util']

"""Change issue_d to date"""

#loan['issue_d'] = pd.to_datetime(loan['issue_d'], format="%b-%Y")
#loan['issue_d']

"""Change address state to object"""

df2['addr_state']=df2['addr_state'].fillna('missing')
df2['addr_state']

pred = gbm_pipeline_hpo.predict_proba(df2)[:,1]

holdout_submission = df2[['id']].copy()
holdout_submission['P_DEFAULT'] = pred
holdout_submission

holdout_submission.to_csv('Final_holdout_submission.csv', index=False)

"""#To HTML"""

# Commented out IPython magic to ensure Python compatibility.
# #Convert to HTML
# %%shell
# jupyter nbconvert --to html "/content/drive/MyDrive/Colab Notebooks/Machine Learning- S3/Project 3- Predict Loan Default/Baldis_Nik_Final_Project.ipynb"
