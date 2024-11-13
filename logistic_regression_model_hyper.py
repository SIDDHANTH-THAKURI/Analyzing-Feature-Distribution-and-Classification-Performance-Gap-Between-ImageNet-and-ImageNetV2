# logistic_regression_model_hyper.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import gc

# Set random seed for reproducibility
np.random.seed(42)

def load_data_in_chunks(file_path, chunksize=5000):
    """
    Generator function to load data in chunks.
    """
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        yield chunk

def prepare_data(chunk):
    """
    Prepare data by separating features and labels.
    """
    # Exclude 'path' and 'label' columns
    features = chunk.drop(['path', 'label'], axis=1)
    # Convert features to float32
    features = features.astype(np.float32)
    # Convert labels to int32
    labels = chunk['label'].astype(np.int32)
    return features, labels

def sample_data_for_tuning(train_file, sample_size=50000):
    """
    Sample a subset of data for hyperparameter tuning.
    """
    print("Sampling data for hyperparameter tuning...")
    sampled_chunks = []
    total_samples = 0
    for chunk in load_data_in_chunks(train_file):
        sampled_chunks.append(chunk)
        total_samples += len(chunk)
        if total_samples >= sample_size:
            break
    data_sample = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()
    X_sample, y_sample = prepare_data(data_sample)
    del data_sample
    gc.collect()
    return X_sample, y_sample

def hyperparameter_tuning(X_sample_scaled, y_sample):
    """
    Perform hyperparameter tuning using Grid Search on a sample of data.
    """
    print("Performing hyperparameter tuning...")
    param_grid = {
        'alpha': [0.0001, 0.001],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['optimal', 'adaptive'],
        'eta0': [0.01, 0.1],
        'max_iter': [1000],
        'tol': [1e-3],
    }
    grid_search = GridSearchCV(
        SGDClassifier(loss='log_loss', random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_sample_scaled, y_sample)
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def fit_scaler_incrementally(train_file, scaler):
    """
    Fit the scaler incrementally on the training data.
    """
    print("Fitting scaler incrementally on full dataset...")
    for chunk in load_data_in_chunks(train_file):
        X_chunk, _ = prepare_data(chunk)
        scaler.partial_fit(X_chunk)
        del X_chunk, chunk
        gc.collect()

def train_logistic_regression_incremental(train_file, model, scaler):
    """
    Train logistic regression model incrementally using SGDClassifier.
    """
    print("Training logistic regression model incrementally with best parameters...")
    for chunk in load_data_in_chunks(train_file):
        X_chunk, y_chunk = prepare_data(chunk)
        X_chunk_scaled = scaler.transform(X_chunk)
        model.partial_fit(X_chunk_scaled, y_chunk, classes=np.arange(1000))
        del X_chunk, y_chunk, X_chunk_scaled, chunk
        gc.collect()

def evaluate_on_test_set(test_file, model, scaler, dataset_name):
    """
    Evaluate the model on a test set.
    """
    y_true = []
    y_pred = []
    print(f"Evaluating on {dataset_name}...")
    for chunk in load_data_in_chunks(test_file):
        X_chunk, y_chunk = prepare_data(chunk)
        X_chunk_scaled = scaler.transform(X_chunk)
        pred_chunk = model.predict(X_chunk_scaled)
        y_true.extend(y_chunk.tolist())
        y_pred.extend(pred_chunk.tolist())
        del X_chunk, y_chunk, X_chunk_scaled, pred_chunk, chunk
        gc.collect()
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred))
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """
    Plot confusion matrix for the given predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', annot=False)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.close()

def main():
    # File paths
    train_file = 'train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
    test1_file = 'val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
    test2_file = 'v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'

    # Sample data for hyperparameter tuning
    X_sample, y_sample = sample_data_for_tuning(train_file)
    scaler_sample = StandardScaler()
    X_sample_scaled = scaler_sample.fit_transform(X_sample)
    del X_sample
    gc.collect()

    # Hyperparameter Tuning
    best_model = hyperparameter_tuning(X_sample_scaled, y_sample)
    del X_sample_scaled, y_sample
    gc.collect()

    # Initialize scaler for full dataset
    scaler_full = StandardScaler()
    fit_scaler_incrementally(train_file, scaler_full)

    # Train model incrementally on full dataset
    best_model.warm_start = True  # Enable warm start for incremental learning
    train_logistic_regression_incremental(train_file, best_model, scaler_full)

    # Evaluate on Test Sets
    y_true_test1, y_pred_test1 = evaluate_on_test_set(test1_file, best_model, scaler_full, "Test Set 1")
    y_true_test2, y_pred_test2 = evaluate_on_test_set(test2_file, best_model, scaler_full, "Test Set 2")

    # Plot confusion matrices
    plot_confusion_matrix(y_true_test1, y_pred_test1, 'Test_Set_1')
    plot_confusion_matrix(y_true_test2, y_pred_test2, 'Test_Set_2')

    # Feature Importance Visualization (Optional)
    if hasattr(best_model, 'coef_'):
        print("Plotting feature importance...")
        importance = np.abs(best_model.coef_).mean(axis=0)
        # Load feature names from first chunk
        chunk = next(load_data_in_chunks(train_file))
        feature_cols = [col for col in chunk.columns if col not in ['path', 'label']]
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        # Plot top 30 features
        plt.figure(figsize=(15, 8))
        sns.barplot(data=feature_importance.head(30), x='importance', y='feature')
        plt.title('Top 30 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        del chunk

if __name__ == "__main__":
    main()
