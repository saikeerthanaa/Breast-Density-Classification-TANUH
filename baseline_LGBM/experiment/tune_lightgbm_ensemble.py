import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

# Set up logging for Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 4,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 255),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**param, n_estimators=2000)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

def main():
    input_file = "final_stratified_ensemble_dataset.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading Ensemble dataset for tuning...")
    df = pd.read_csv(input_file)
    df['label'] = df['density_class'].astype(float).astype(int) - 1
    feat_cols = [c for c in df.columns if 'resnet_feat_' in c or 'effnet_feat_' in c]
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    X_train, y_train = train_df[feat_cols], train_df['label']
    X_val, y_val = val_df[feat_cols], val_df['label']
    X_test, y_test = test_df[feat_cols], test_df['label']

    print(f"Starting Optuna optimization (50 trials)...")
    study = optuna.create_study(direction='maximize')
    
    def callback(study, trial):
        print(f"Trial {trial.number} finished with value: {trial.value:.4f} and parameters: {trial.params}")
        print(f"Best value so far: {study.best_value:.4f}")

    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50, callbacks=[callback])

    print("\nOptimization complete.")
    print(f"Best trial accuracy: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Train final model with best params
    print("\nTraining final model with best parameters...")
    best_params = study.best_params
    best_params.update({
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 4,
        'random_state': 42,
        'n_jobs': -1,
        'n_estimators': 2000
    })
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
    )

    print("\nEvaluating on Test Set...")
    y_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save results
    with open('Ensemble_Tuned_Results.md', 'w') as f:
        f.write("# Tuned Ensemble (ResNet50 + EffNet-B7) Results\n\n")
        f.write(f"- **Final Test Accuracy:** {test_acc:.4f}\n")
        f.write(f"- **Best Val Accuracy:** {study.best_value:.4f}\n\n")
        f.write("## Best Parameters\n")
        for k, v in study.best_params.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Classification Report\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
