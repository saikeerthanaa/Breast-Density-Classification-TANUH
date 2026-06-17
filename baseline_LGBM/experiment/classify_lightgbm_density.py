import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

def classify():
    features_file = "patient_features.csv"
    clinical_file = "data/EMBED/tables/EMBED_OpenData_clinical_reduced.csv"
    
    if not os.path.exists(features_file):
        print(f"Error: {features_file} not found. Please run feature extraction first.")
        return

    print("Loading features and labels...")
    df_feat = pd.read_csv(features_file)
    df_clinical = pd.read_csv(clinical_file)
    
    # Filter for patients that have a valid tissue density label (1, 2, 3, 4)
    # Dropping 5.0 and NaN based on analysis
    valid_densities = [1.0, 2.0, 3.0, 4.0]
    df_clinical = df_clinical[df_clinical['tissueden'].isin(valid_densities)]
    
    # Get max density per patient (or mean, though density is categorical)
    # Using max as a simple representative
    df_labels = df_clinical.groupby('empi_anon')['tissueden'].max().reset_index()
    df_labels.columns = ['patient_id', 'label']
    
    # Ensure patient_id is string for merging
    df_feat['patient_id'] = df_feat['patient_id'].astype(str)
    df_labels['patient_id'] = df_labels['patient_id'].astype(str)
    
    # Merge
    data = pd.merge(df_feat, df_labels, on='patient_id', how='inner')
    
    if data.empty:
        print("No matching patients with labels found!")
        return
    
    print(f"Dataset size after merging: {len(data)}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    # Features are columns other than patient_id and label
    X = data.drop(['patient_id', 'label'], axis=1)
    y = data['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        objective='multiclass',
        num_class=len(le.classes_),
        random_state=42,
        importance_type='gain',
        device='gpu' if os.environ.get('USE_GPU') == '1' else 'cpu'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
    )
    
    y_pred = model.predict(X_test)
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    classify()
