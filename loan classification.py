from seaborn import set_style
from sklearn import datasets
from sklearn import metrics
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
# Import the dataset
loan_data = pd.read_csv('F:\\Kaggle\\loan approval\\loan_data.csv')
loan_data.shape
loan_data.head()
#Inspect the dataset
loan_data.info()
loan_data['person_age'] = loan_data['person_age'].astype('int64')
loan_data.describe()
#Check unique values in character columns
for col in loan_data.select_dtypes(include='object').columns:
    print(f"Unique values in '{col}': {loan_data[col].unique()}")
#Check for missing values
loan_data.isna().sum()
sns.boxplot(data=loan_data, x='person_age')
plt.xlabel('Person Age')
# Check for outliers in person_age
def rosner_test(data, alpha=0.05):
    n = len(data)
    k = 10  # Number of outliers to detect
    data_sorted = np.sort(data)
    # Calculate the mean and standard deviation
    mean = np.mean(data_sorted)
    std_dev = np.std(data_sorted, ddof=1)  # Sample standard deviation
    # Check for outliers
    outliers = []
    for i in range(k):
        # Calculate the Rosner statistic
        rosner_stat = (data_sorted[-(i + 1)] - mean) / std_dev
        outliers.append((data_sorted[-(i + 1)], rosner_stat))
    return outliers
# Perform Rosner's Test
outlier_results = rosner_test(loan_data['person_age'])
print("Outliers detected with Rosner's Test:")
for value, stat in outlier_results:
    print(f"Value: {value}, Rosner Statistic: {stat}")
#Statistical summary and tests
loan_data = loan_data[loan_data['person_age'] <= 80]
loan_data.describe()
sns.histplot(data=loan_data, x='person_income', bins = 100)
plt.xlabel('Person Income')
sns.histplot(data=loan_data, x='loan_amnt', bins = 100)
plt.xlabel('Loan Amount')
#ANOVA test for education groups, target variable is person_income
from scipy import stats
from scipy.stats import f_oneway
groups = loan_data.groupby('person_education')['person_income'].apply(list)
f_statistic, p_value = stats.f_oneway(*groups)
print("F-statistic:", f_statistic)
print("p-value:", p_value)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences in income based on education.")
else:
    print("Fail to reject the null hypothesis: No significant differences in income based on education.")
#descriptive table
mean_income_table = loan_data.groupby('person_education')['person_income'].mean().reset_index()
mean_income_table.columns = ['Person Education', 'Average Income']
mean_income_table
#Perfom point biserial correlation test for loan_status and other continuous variables
from scipy.stats import pointbiserialr
numeric_vars = ['person_income', 'person_age','person_emp_exp','loan_amnt','cb_person_cred_hist_length','credit_score']
correlations = {}
for var in numeric_vars:
    corr, p_value = pointbiserialr(loan_data['loan_status'], loan_data[var])
    correlations[var] = {'correlation': corr, 'p-value': p_value}

correlation_df = pd.DataFrame(correlations).T
print(correlation_df)
#ch square test for categorical variables
from scipy.stats import chi2_contingency
categorical_vars = ['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
results = {}
for var in categorical_vars:
    contingency_table = pd.crosstab(loan_data[var], loan_data['loan_status'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    results[var] = {'chi2_statistic': chi2, 'p_value': p}
results_df = pd.DataFrame(results).T
print(results_df)
#Check target variable
class_counts = loan_data['loan_status'].value_counts()

# Bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Class Distribution of loan_status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
# Pie plot
plt.figure(figsize=(6, 6))
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Loan Status Distribution')
plt.show()
#voting classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Define the preprocessing steps
X = loan_data.drop('loan_status', axis=1)
y = loan_data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = ['person_age','person_emp_exp','cb_person_cred_hist_length','credit_score'] 
log_features = ['person_income','loan_amnt']
# Define transformation pipelines
log_pipeline = Pipeline([
    ('log', FunctionTransformer(np.log, validate=True)),
    ('scaler', StandardScaler())])
num_pipeline = Pipeline([
    ('scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        ('log', log_pipeline, log_features),
        ('num', num_pipeline, numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
# Define the voting classifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# Create classifiers
dt_classifier = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=6)
lr_classifier = LogisticRegression(class_weight='balanced', random_state=42, max_iter=200)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Create a voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('decision_tree', dt_classifier),
    ('logistic_regression', lr_classifier),
    ('random_forest', rf_classifier)
], voting='soft')

# Initialize RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)

# Create a pipeline with undersampling
pipeline = Pipeline(steps=[
    ('under_sampler', undersampler),
    ('preprocessor', preprocessor),
    ('classifier', voting_classifier)
])
#MLflow
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")
print("Tracking URI now =", mlflow.get_tracking_uri())
try:
    with mlflow.start_run():
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        # Predict
        y_pred = pipeline.predict(X_test)
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)  # để log được f1, precision
        # Log params
        mlflow.log_param("model_type", "VotingClassifier")
        mlflow.log_param("voting", "soft")
        mlflow.log_param("dt_max_depth", 6)
        mlflow.log_param("rf_n_estimators", 100)
        mlflow.log_param("sampling_strategy", "RandomUnderSampler")
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        # Log model
        mlflow.sklearn.log_model(pipeline, "loan_voting_model")
except Exception as e:
    print("Error during MLflow run:", e)
    mlflow.end_run(status="FAILED")  # Mark failed nếu có bug
else:
    mlflow.end_run(status="FINISHED") 
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
#auc roc
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# Make predictions (probability estimates)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # get probabilities for the positive class
# Calculate AUC
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC Score: {auc_score:.4f}")
# Optional: Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#calibrate
from sklearn.calibration import CalibratedClassifierCV
calibrated_clf = CalibratedClassifierCV(estimator=voting_classifier, method='isotonic', cv=5)
pipeline_calibrated = Pipeline(steps=[
    ('undersampler', undersampler),
    ('preprocessor', preprocessor),
    ('classifier', calibrated_clf)
])
pipeline_calibrated.fit(X_train, y_train)
y_proba = pipeline_calibrated.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Isotonic Calibrated)')
plt.legend()

#PR curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'AP = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Isotonic Calibrated)')
plt.legend()

plt.tight_layout()
plt.show()
#brier score
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, y_proba)
print(f"Brier Score: {brier:.4f}")
#tuning
from sklearn.metrics import precision_recall_curve

# Get predicted probabilities for class 1
y_proba = pipeline.predict_proba(X_test)[:, 1]
# Evaluate PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Compute F1-score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Find the threshold with the best F1-score
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print(f"✅ Best threshold (F1): {best_threshold:.3f}")
print(f"📈 Best F1-score: {best_f1:.3f}")

# Plot to find trade-off
import matplotlib.pyplot as plt
plt.plot(thresholds, precision[1:], label='Precision')
plt.plot(thresholds, recall[1:], label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.title('Precision-Recall Trade-off')
plt.show()

# Choose a threshold, e.g., 0.6
custom_threshold = 0.709
y_pred_custom = (y_proba >= custom_threshold).astype(int)
# Apply the best threshold for final predictions
y_pred = (y_proba >= best_threshold).astype(int)

# Evaluate again
from sklearn.metrics import classification_report
print("📊 Report @ threshold = 0.5")
print(classification_report(y_test, (y_proba >= 0.5).astype(int)))

print("📊 Report @ threshold = best F1")
print(classification_report(y_test, y_pred))
