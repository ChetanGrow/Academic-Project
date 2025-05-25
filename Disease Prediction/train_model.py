import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Define scaling ranges (broad physiological ranges)
scaling_ranges = {
    "RBC": (3.5, 6.0),
    "HGB": (10.0, 18.0),
    "HCT": (30.0, 55.0),
    "MCV": (70.0, 110.0),
    "MCHC": (28.0, 38.0),
    "RDW": (10.0, 20.0),
    "PLT": (100, 600),
    "WBC": (3.0, 15.0)
}

# Normal value ranges for abnormal flags (standard reference ranges)
normal_ranges = {
    "RBC": (4.5, 5.5),
    "HGB": (12.0, 16.0),
    "HCT": (36.0, 48.0),
    "MCV": (80.0, 100.0),
    "MCHC": (32.0, 36.0),
    "RDW": (11.5, 14.5),
    "PLT": (150, 450),
    "WBC": (4.0, 11.0)
}

# Reverse-normalize data (dataset is in [0, 1], map to physiological ranges)
def reverse_normalize_data(df, scaling_ranges):
    df_reversed = df.copy()
    for col, (min_val, max_val) in scaling_ranges.items():
        if col in df.columns:
            df_reversed[col] = df[col] * (max_val - min_val) + min_val
            print(f"Reverse-Normalized {col} - Min: {df_reversed[col].min()}, Max: {df_reversed[col].max()}")
    return df_reversed

# Normalize data to [0, 1]
def normalize_data(df, scaling_ranges):
    df_normalized = df.copy()
    for col, (min_val, max_val) in scaling_ranges.items():
        if col in df.columns:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            df_normalized[col] = df_normalized[col].clip(0, 1)
            print(f"Normalized {col} - Min: {df_normalized[col].min()}, Max: {df_normalized[col].max()}")
    return df_normalized

# Create abnormal feature columns
def create_abnormal_features(df, normal_ranges):
    df_abnormal = df.copy()
    for col, (low, high) in normal_ranges.items():
        if col in df.columns:
            df_abnormal[f"{col}_abnormal"] = ((df[col] < low) | (df[col] > high)).astype(int)
            print(f"{col}_abnormal distribution: {df_abnormal[f'{col}_abnormal'].value_counts().to_dict()}")
    return df_abnormal

# Load dataset
data = pd.read_csv('Blood_samples_dataset_balanced.csv')

# Print available columns to understand structure
print("\nAvailable Columns in Dataset:")
print(data.columns.tolist())

# Rename columns for consistency
fbc_columns = {
    'Hemoglobin': 'HGB',
    'Platelets': 'PLT',
    'White Blood Cells': 'WBC',
    'Red Blood Cells': 'RBC',
    'Hematocrit': 'HCT',
    'Mean Corpuscular Volume': 'MCV',
    'Mean Corpuscular Hemoglobin Concentration': 'MCHC',
    'Red Cell Distribution Width': 'RDW'
}
data.rename(columns=fbc_columns, inplace=True)

# Common features between dataset and input
common_features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCHC', 'RDW', 'PLT', 'WBC']

# Filter out missing columns
selected_features = [col for col in common_features if col in data.columns]
print(f"\nSelected Features Found in Dataset: {selected_features}")

# Drop rows with missing values in selected features or target
data = data.dropna(subset=selected_features + ['Disease'])

# Define features (X) and target (y)
X = data[selected_features]
y = data['Disease']

# Print raw data ranges for debugging
print("\nRaw Data Ranges:")
for col in selected_features:
    print(f"{col} - Min: {X[col].min()}, Max: {X[col].max()}")

# Reverse-normalize the data to physiological ranges
X_reversed = reverse_normalize_data(X, scaling_ranges)

# Add abnormal features on the reverse-normalized (physiological) data
X_with_abnormal = create_abnormal_features(X_reversed, normal_ranges)

# Normalize features to [0, 1]
X_normalized = normalize_data(X_with_abnormal, scaling_ranges)

# Update selected features to include abnormal flags
all_features = selected_features + [f"{col}_abnormal" for col in selected_features if col in normal_ranges]

# Print class distribution before balancing
print("\nClass Distribution Before SMOTE:")
print(y.value_counts())

# Create holdout set before SMOTE
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X_normalized, y, test_size=0.1, random_state=42, stratify=y)

# Apply SMOTE for class balance (partial oversampling)
smote = SMOTE(sampling_strategy='not majority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)


print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Standardize features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_holdout_scaled = scaler.transform(X_holdout)

# Split resampled data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
print(cm_df)

# Evaluate on holdout set
y_holdout_pred = model.predict(X_holdout_scaled)
print(f"\nHoldout Accuracy: {accuracy_score(y_holdout, y_holdout_pred):.4f}")

# Classification report for holdout set
print("\nClassification Report (Holdout Set):")
print(classification_report(y_holdout, y_holdout_pred))

# Stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_resampled_scaled, y_resampled, cv=skf)
print(f"\nCross-Validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance (native for RandomForest)
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances (Random Forest):")
print(feature_importance)

# Check for data leakage (feature-target correlations with one-hot encoded target)
print("\nFeature-Target Correlations (with one-hot encoded target):")
y_one_hot = pd.get_dummies(y)
X_temp_df = pd.DataFrame(X, columns=selected_features)
correlations = X_temp_df.corrwith(y_one_hot.iloc[:, 0], axis=0)
for col in y_one_hot.columns[1:]:
    correlations = pd.concat([correlations, X_temp_df.corrwith(y_one_hot[col], axis=0)])

print(correlations.groupby(level=0).mean().sort_values(ascending=False))

# Inspect SMOTE resampled data
print("\nResampled Data Stats (Before Scaling):")
print(pd.DataFrame(X_resampled, columns=all_features).describe())

# Save model, feature names, and scaler
joblib.dump((model, all_features, scaler), 'disease_prediction_model.joblib')
print("\nModel saved to 'disease_prediction_model.joblib'")