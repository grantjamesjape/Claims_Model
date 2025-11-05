import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import os
import warnings
warnings.filterwarnings("ignore") 

# define standard folder names based on your GitHub repository structure
DATA_DIR = 'data'
CHARTS_DIR = 'charts'
TRANSFORMED_DATA_FILENAME = 'insurance.csv' 

# create output directories
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
print("Setup: Output directories created successfully.")

# import data
data_path = os.path.join(DATA_DIR, 'insurance.csv')
df = pd.read_csv(data_path)

# set plotting style
sns.set_style("whitegrid")

# EDA ===============================================================

# descriptive stats
stats = round(df['charges'].describe(), 2)
stats_df = stats.to_frame().reset_index()
stats_df.columns = ['Statistics', 'Value']

stats_path = os.path.join(DATA_DIR, "charges_summary_stats.csv")
stats_df.to_csv(stats_path, index=False)

# scatterplots for continuous variables
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# left plot: age vs charges
sns.scatterplot(x='age', y='charges', data=df, ax=axes[0], alpha=0.6, hue='smoker', palette='Set1')
axes[0].set_title('Age vs. Charges (by Smoker Status)', fontsize=14)
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Charges', fontsize=12)
axes[0].legend(title='Smoker')

# right plot: bmi vs charges
sns.scatterplot(x='bmi', y='charges', data=df, ax=axes[1], alpha=0.6, hue='smoker', palette='Set1')
axes[1].set_title('BMI vs. Charges (by Smoker Status)', fontsize=14)
axes[1].set_xlabel('BMI', fontsize=12)
axes[1].set_ylabel('Charges', fontsize=12)
axes[1].legend(title='Smoker')

plt.tight_layout()

# save to charts folder
output_path_scat = os.path.join(CHARTS_DIR, 'scatterplots.png')
plt.savefig(output_path_scat)
plt.close()

# boxplots for categorical variables
categorical_features = ['smoker', 'region', 'sex']

fig, axes = plt.subplots(1, 3, figsize=(20,9))

for i, feature in enumerate(categorical_features):
    ax = axes[i]
    sns.boxplot(x=feature, y='charges', data=df, ax=ax, palette="viridis", hue=feature, legend=False)
    ax.set_title(f'Charges vs. {feature.capitalize()}', fontsize=14)
    ax.set_xlabel(feature.capitalize(), fontsize=12)
    ax.set_ylabel('Charges', fontsize=12)

# save to charts folder
plt.tight_layout()
output_path_box = os.path.join(CHARTS_DIR, 'boxplots.png')
plt.savefig(output_path_box)
plt.close()

# distrubution of charges
df['log_charges'] = np.log(df['charges'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original Distribution of Charges
sns.histplot(df['charges'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Charges (Original)', fontsize=14)
axes[0].set_xlabel('Charges', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

# Log-Transformed Distribution
sns.histplot(df['log_charges'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Distribution of Log-Transformed Charges', fontsize=14)
axes[1].set_xlabel('Log(Charges)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)

# save to charts folder
plt.tight_layout()
output_path_dis = os.path.join(CHARTS_DIR, 'distributions.png')
plt.savefig(output_path_dis)
plt.close()

# corr plot
df['log_charges'] = np.log(df['charges'])

categorical_features = ['sex', 'smoker', 'region']

# one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int)

# Select all numerical and newly encoded columns, dropping the original 'charges'
correlation_matrix = df_encoded.drop(columns=['charges']).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title('Feature Correlation Heatmap', fontsize=16)

# save to charts folder
plt.tight_layout()
output_path_corr = os.path.join(CHARTS_DIR, 'corr.png')
plt.savefig(output_path_corr)
plt.close()

# grouped stats
grouped_smoker = df.groupby('smoker')['charges'].describe()
grouped_smoker = round(grouped_smoker, 2)
grouped_smoker_T_path = os.path.join(DATA_DIR, 'grouped_charges_by_smoker_T.csv')
grouped_smoker.T.to_csv(grouped_smoker_T_path)


# feature engineering ================================================================================
# create a binary flag where 1 means BMI is 30 or greater
df['is_obese'] = (df['bmi'] >= 30).astype(int)

# interaction term: smoker_obese_interaction
df['smoker_obese_interaction'] = ((df['smoker'] == 'yes') & (df['is_obese'] == 1)).astype(int)

# derived feature: age groups
bins = [18, 30, 45, 65, 100]
labels = ['Young_Adult', 'Middle_Adult', 'Senior_Adult', 'Elderly']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Display the results for verification
print("Data Head with New Engineered Features")
print(df[['age', 'bmi', 'smoker', 'is_obese', 'smoker_obese_interaction', 'age_group', 'charges']].head(10))

print("\nMean Charges by Smoker-Obese Interaction")
interaction_charges = df.groupby('smoker_obese_interaction')['charges'].mean()
print(interaction_charges.apply(lambda x: f"${x:,.2f}"))


# Modeling ===============================================================================================
# get dummies
df = pd.get_dummies(df, drop_first=True)

# split train test
X = df.drop(["charges", "log_charges"], axis=1)
y = df['charges']
y_log = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# select and train models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

pd.DataFrame(results).T

# tune model 
gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 500],     # number of boosting stages
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # how much each tree corrects the previous one
    'max_depth': [3, 4, 5, 6],                # tree depth (model complexity)
    'min_samples_split': [2, 5, 10],          # minimum samples to split a node
    'subsample': [0.8, 1.0]                   # fraction of samples used per tree
}

grid = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # minimize MSE
    cv=5,                              # 5-fold cross-validation
    n_jobs=-1,                         # use all cores
    verbose=2
)

grid.fit(X_train, y_train)
print("Best parameters found:", grid.best_params_)
print("Best CV score (negative MSE):", grid.best_score_)

# retrain using best parameters
best_gbr = grid.best_estimator_
best_gbr.fit(X_train, y_train)

# fitting, predictions and evaluation
y_train_pred = best_gbr.predict(X_train)
y_test_pred  = best_gbr.predict(X_test)

# metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# results
print(f"Train MAE:  {train_mae:.4f},  Test MAE:  {test_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R²:   {train_r2:.4f},   Test R²:   {test_r2:.4f}")

# actual vs predicted
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Charges")

# save to charts folder
plt.tight_layout()
output_path_fit = os.path.join(CHARTS_DIR, 'fit.png')
plt.savefig(output_path_fit)
plt.close()


# feature importances
importances = pd.Series(best_gbr.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar')

# convert y target back to charges ($)
y_pred_dollar = np.exp(y_test_pred)
y_actual_dollar = np.exp(y_test)

# evaluation in true charges
mae_dollar = mean_absolute_error(y_actual_dollar, y_pred_dollar)
rmse_dollar = np.sqrt(mean_squared_error(y_actual_dollar, y_pred_dollar))
r2_dollar = r2_score(y_actual_dollar, y_pred_dollar)

print(f"MAE: ${mae_dollar:,.2f}")
print(f"RMSE: ${rmse_dollar:,.2f}")
print(f"R²: {r2_dollar:.3f}")










