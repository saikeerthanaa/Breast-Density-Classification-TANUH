import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

def main():
    input_file = "final_stratified_effnet_b7_dataset.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading EfficientNet-B7 dataset...")
    df = pd.read_csv(input_file)
    
    # Map density_class to 0-3
    df['label'] = df['density_class'].astype(float).astype(int) - 1
    
    # Identify feature columns (2560 for EffNet-B7)
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    X_train, y_train = train_df[feat_cols], train_df['label']
    X_val, y_val = val_df[feat_cols], val_df['label']
    X_test, y_test = test_df[feat_cols], test_df['label']
    
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    
    print("Training LightGBM model on EfficientNet-B7 features...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        objective='multiclass',
        num_class=4,
        random_state=42,
        importance_type='gain',
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    print("\nEvaluating on Test Set...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy (EffNet-B7): {acc:.4f}")
    
    target_names = ['Class 1 (Fatty)', 'Class 2 (Scattered)', 'Class 3 (Heterogeneous)', 'Class 4 (Dense)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
