# sgd_classifier_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
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
    features = chunk.drop(['path', 'label'], axis=1)
    # Convert features to float32
    features = features.astype(np.float32)
    # Convert labels to int32
    labels = chunk['label'].astype(np.int32)
    return features, labels

def fit_scaler_incrementally(train_file, scaler):
    """
    Fit the scaler incrementally on the training data.
    """
    print("Fitting scaler incrementally...")
    for chunk in load_data_in_chunks(train_file):
        X_chunk, _ = prepare_data(chunk)
        scaler.partial_fit(X_chunk)
        del X_chunk, chunk
        gc.collect()

def train_sgd_incremental(train_file, model, scaler):
    """
    Train SGDClassifier incrementally.
    """
    print("Training SGDClassifier incrementally...")
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
        y_true.extend(y_chunk)
        y_pred.extend(pred_chunk)
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

    # Initialize scaler and model
    scaler = StandardScaler()
    model = SGDClassifier(random_state=42)

    # Fit scaler incrementally
    fit_scaler_incrementally(train_file, scaler)

    # Train model incrementally
    train_sgd_incremental(train_file, model, scaler)

    # Evaluate on Test Sets
    y_true_test1, y_pred_test1 = evaluate_on_test_set(test1_file, model, scaler, "Test Set 1")
    y_true_test2, y_pred_test2 = evaluate_on_test_set(test2_file, model, scaler, "Test Set 2")

    # Plot confusion matrices
    plot_confusion_matrix(y_true_test1, y_pred_test1, 'Test_Set_1')
    plot_confusion_matrix(y_true_test2, y_pred_test2, 'Test_Set_2')

    # t-SNE Visualization (optional, similar to previous scripts)
    # You can add t-SNE visualization here if desired.

if __name__ == "__main__":
    main()
